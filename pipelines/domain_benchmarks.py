import argparse
import csv
import json
import math
import sqlite3
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import pipeline


KAGGLE_SOCIAL_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/mdismielhossenabir/sentiment-analysis"
KAGGLE_SOCIAL_ZIP_NAME = "mdismielhossenabir_sentiment-analysis.zip"
KAGGLE_SOCIAL_CSV_NAME = "sentiment_analysis.csv"

KAGGLE_REVIEWS_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/dolbokostya/test-dataset"
KAGGLE_REVIEWS_ZIP_NAME = "dolbokostya_test-dataset.zip"
KAGGLE_REVIEWS_CSV_NAME = "2.5m-reviews-dataset.csv"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    model: str | None = None
    repo: str | None = None
    file: str | None = None
    domain: str = "general"


MODEL_SPECS = [
    ModelSpec(
        name="Gemma 4 E2B",
        kind="llm",
        repo="unsloth/gemma-4-E2B-it-GGUF",
        file="gemma-4-E2B-it-Q4_0.gguf",
        domain="general",
    ),
    ModelSpec(
        name="Qwen3.5 0.8B Q4",
        kind="llm",
        repo="unsloth/Qwen3.5-0.8B-GGUF",
        file="Qwen3.5-0.8B-Q4_0.gguf",
        domain="general",
    ),
    ModelSpec(
        name="Specialized: BERTweet Sentiment",
        kind="transformer",
        model="finiteautomata/bertweet-base-sentiment-analysis",
        domain="social",
    ),
    ModelSpec(
        name="Specialized: Twitter-RoBERTa",
        kind="transformer",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        domain="social",
    ),
    ModelSpec(
        name="Specialized: Reviews-SST2",
        kind="transformer",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        domain="reviews",
    ),
]


def normalize_label(raw_label: str) -> str:
    label = (raw_label or "").strip().lower()
    if label in {"label_0", "0", "negative", "neg", "1 star", "1 stars", "2 stars"}:
        return "NEGATIVE"
    if label in {"label_1", "1", "neutral", "neu", "3 stars"}:
        return "NEUTRAL"
    if label in {"label_2", "2", "positive", "pos", "4 stars", "5 stars"}:
        return "POSITIVE"
    if "pos" in label:
        return "POSITIVE"
    if "neg" in label:
        return "NEGATIVE"
    if "neu" in label:
        return "NEUTRAL"
    return "UNKNOWN"


def parse_llm_label(raw_output: str) -> str:
    up = raw_output.upper()
    for label in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
        if label in up:
            return label
    return normalize_label(raw_output)


def build_llm(repo_id: str, filename: str) -> Llama:
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Llama(model_path=model_path, n_ctx=4096, verbose=False)


def llm_predict(llm: Llama, text: str) -> tuple[str, str]:
    prompt = (
        "Classify the sentiment of this text as POSITIVE, NEGATIVE, or NEUTRAL. "
        "Reply with exactly one label: POSITIVE, NEGATIVE, or NEUTRAL.\n\n"
        f"Text: {text}\n"
        "Label:"
    )
    response = llm.create_completion(prompt=prompt, max_tokens=6, temperature=0.0)
    raw_output = response["choices"][0]["text"].strip()
    return parse_llm_label(raw_output), raw_output


def download_kaggle_zip(download_url: str, zip_name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / zip_name
    if not zip_path.exists():
        urllib.request.urlretrieve(download_url, zip_path)
    return zip_path


def load_instagram_rows(max_samples: int) -> list[dict]:
    dataset = load_dataset("pgurazada1/instagram-comments-sentiment", split="test")
    rows = []
    for i, row in enumerate(dataset):
        rows.append(
            {
                "sample_index": i,
                "source_id": str(row["comment_id"]),
                "text": row["text"],
                "gold_label": normalize_label(row["sentiment_label_rater1"]),
            }
        )
        if max_samples > 0 and len(rows) >= max_samples:
            break
    return rows


def load_kaggle_social_rows(max_samples: int, cache_dir: Path) -> list[dict]:
    zip_path = download_kaggle_zip(
        download_url=KAGGLE_SOCIAL_DOWNLOAD_URL,
        zip_name=KAGGLE_SOCIAL_ZIP_NAME,
        cache_dir=cache_dir,
    )
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(KAGGLE_SOCIAL_CSV_NAME) as csv_file:
            reader = csv.DictReader((line.decode("utf-8") for line in csv_file))
            for i, row in enumerate(reader):
                rows.append(
                    {
                        "sample_index": i,
                        "source_id": str(i),
                        "text": row.get("text", ""),
                        "gold_label": normalize_label(row.get("sentiment", "")),
                    }
                )
                if max_samples > 0 and len(rows) >= max_samples:
                    break
    return rows


def _pick_first_present(row: dict, candidates: list[str]) -> str | None:
    lowered = {key.lower(): key for key in row.keys()}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None:
            return key
    return None


def load_kaggle_reviews_rows(max_samples: int, cache_dir: Path) -> list[dict]:
    zip_path = download_kaggle_zip(
        download_url=KAGGLE_REVIEWS_DOWNLOAD_URL,
        zip_name=KAGGLE_REVIEWS_ZIP_NAME,
        cache_dir=cache_dir,
    )
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(KAGGLE_REVIEWS_CSV_NAME) as csv_file:
            reader = csv.DictReader((line.decode("utf-8") for line in csv_file))
            first = next(reader, None)
            if first is None:
                return rows

            text_key = _pick_first_present(first, ["text", "review", "content", "sentence"])
            label_key = _pick_first_present(first, ["sentiment", "label", "stars", "rating", "score"])
            if text_key is None or label_key is None:
                raise ValueError("Could not infer text/label columns from dolbokostya/test-dataset CSV")

            seed_rows = [first]
            for row in seed_rows:
                rows.append(
                    {
                        "sample_index": len(rows),
                        "source_id": str(len(rows)),
                        "text": row.get(text_key, ""),
                        "gold_label": normalize_label(row.get(label_key, "")),
                    }
                )

            for row in reader:
                rows.append(
                    {
                        "sample_index": len(rows),
                        "source_id": str(len(rows)),
                        "text": row.get(text_key, ""),
                        "gold_label": normalize_label(row.get(label_key, "")),
                    }
                )
                if max_samples > 0 and len(rows) >= max_samples:
                    break
    return rows


def evaluate_transformer(model_name: str, rows: list[dict]) -> tuple[float, float, list[dict]]:
    classifier = pipeline("sentiment-analysis", model=model_name)
    total_correct = 0
    total_latency = 0.0
    predictions = []

    for row in rows:
        start = time.perf_counter()
        output = classifier(row["text"], truncation=True)[0]
        latency = time.perf_counter() - start
        predicted = normalize_label(output.get("label", ""))
        is_correct = int(predicted == row["gold_label"])
        total_correct += is_correct
        total_latency += latency

        predictions.append(
            {
                "sample_index": row["sample_index"],
                "source_id": row["source_id"],
                "text": row["text"],
                "gold_label": row["gold_label"],
                "predicted_label": predicted,
                "is_correct": is_correct,
                "latency_seconds": latency,
                "raw_prediction": output,
            }
        )

    accuracy = (total_correct / len(rows)) if rows else 0.0
    avg_latency = (total_latency / len(rows)) if rows else 0.0
    return accuracy, avg_latency, predictions


def evaluate_llm(repo: str, filename: str, rows: list[dict]) -> tuple[float, float, list[dict]]:
    llm = build_llm(repo_id=repo, filename=filename)
    total_correct = 0
    total_latency = 0.0
    predictions = []

    for row in rows:
        start = time.perf_counter()
        predicted, raw_output = llm_predict(llm, row["text"])
        latency = time.perf_counter() - start
        is_correct = int(predicted == row["gold_label"])
        total_correct += is_correct
        total_latency += latency

        predictions.append(
            {
                "sample_index": row["sample_index"],
                "source_id": row["source_id"],
                "text": row["text"],
                "gold_label": row["gold_label"],
                "predicted_label": predicted,
                "is_correct": is_correct,
                "latency_seconds": latency,
                "raw_prediction": raw_output,
            }
        )

    accuracy = (total_correct / len(rows)) if rows else 0.0
    avg_latency = (total_latency / len(rows)) if rows else 0.0
    return accuracy, avg_latency, predictions


def _ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_kind TEXT NOT NULL,
                model_domain TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                average_latency_seconds REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                run_id TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                sample_index INTEGER NOT NULL,
                source_id TEXT NOT NULL,
                text TEXT NOT NULL,
                gold_label TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                latency_seconds REAL NOT NULL,
                raw_prediction TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_results_to_sqlite(
    db_path: Path,
    created_at: str,
    dataset_name: str,
    spec: ModelSpec,
    accuracy: float,
    average_latency: float,
    predictions: list[dict],
) -> str:
    run_id = f"{created_at}::{dataset_name}::{spec.name}"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, created_at, dataset_name, model_name, model_kind, model_domain,
                sample_count, accuracy, average_latency_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                dataset_name,
                spec.name,
                spec.kind,
                spec.domain,
                len(predictions),
                accuracy,
                average_latency,
            ),
        )
        conn.executemany(
            """
            INSERT INTO predictions (
                run_id, dataset_name, model_name, sample_index, source_id, text,
                gold_label, predicted_label, is_correct, latency_seconds, raw_prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    dataset_name,
                    spec.name,
                    row["sample_index"],
                    row["source_id"],
                    row["text"],
                    row["gold_label"],
                    row["predicted_label"],
                    row["is_correct"],
                    row["latency_seconds"],
                    json.dumps(row["raw_prediction"], ensure_ascii=False),
                )
                for row in predictions
            ],
        )
        conn.commit()
    return run_id


def build_bar_chart(
    dataset_names: list[str],
    values_by_model: dict[str, list[float]],
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    x_positions = list(range(len(dataset_names)))
    model_names = list(values_by_model.keys())
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(model_names):
        offsets = [x + (i - (len(model_names) - 1) / 2) * bar_width for x in x_positions]
        ax.bar(offsets, values_by_model[model_name], width=bar_width, label=model_name)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(dataset_names, rotation=15)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_correctness_grid(dataset_name: str, matrix_rows: list[tuple[str, list[int]]], output_path: Path) -> None:
    if not matrix_rows:
        return

    max_samples = max(len(row[1]) for row in matrix_rows)
    fig_height = max(3, len(matrix_rows) * 0.8)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for y, (model_name, correctness) in enumerate(matrix_rows):
        for x, value in enumerate(correctness):
            color = "#2ca02c" if value == 1 else "#d62728"
            ax.scatter(x, y, marker="s", s=20, color=color)

    ax.set_yticks(list(range(len(matrix_rows))))
    ax.set_yticklabels([row[0] for row in matrix_rows])
    ax.set_xticks(list(range(0, max_samples, max(1, math.ceil(max_samples / 20)))))
    ax.set_xlabel("Sample index")
    ax.set_title(f"Per-item correctness grid: {dataset_name}")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _load_selected_datasets(selected: list[str], social_samples: int, review_samples: int, cache_dir: Path) -> dict[str, list[dict]]:
    datasets: dict[str, list[dict]] = {}
    if "instagram" in selected:
        datasets["Instagram Comments (test)"] = load_instagram_rows(max_samples=social_samples)
    if "kaggle_social" in selected:
        datasets["Kaggle Sentiment Analysis (mdismielhossenabir)"] = load_kaggle_social_rows(
            max_samples=social_samples,
            cache_dir=cache_dir,
        )
    if "kaggle_reviews" in selected:
        datasets["Kaggle Reviews (dolbokostya subset)"] = load_kaggle_reviews_rows(
            max_samples=review_samples,
            cache_dir=cache_dir,
        )
    return datasets


def run_all_domain_benchmarks(
    social_samples: int,
    review_samples: int,
    output_dir: Path,
    cache_dir: Path,
    sqlite_path: Path,
    selected_datasets: list[str],
) -> dict:
    datasets = _load_selected_datasets(
        selected=selected_datasets,
        social_samples=social_samples,
        review_samples=review_samples,
        cache_dir=cache_dir,
    )

    _ensure_db(sqlite_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"datasets": {}, "models": [spec.name for spec in MODEL_SPECS]}
    accuracy_for_plot = {spec.name: [] for spec in MODEL_SPECS}
    latency_for_plot = {spec.name: [] for spec in MODEL_SPECS}
    created_at = datetime.now(UTC).isoformat()

    for dataset_name, rows in datasets.items():
        summary["datasets"][dataset_name] = {"samples": len(rows), "results": {}}
        grid_rows: list[tuple[str, list[int]]] = []

        for spec in MODEL_SPECS:
            if spec.kind == "transformer":
                accuracy, avg_latency, predictions = evaluate_transformer(spec.model or "", rows)
            else:
                accuracy, avg_latency, predictions = evaluate_llm(spec.repo or "", spec.file or "", rows)

            save_results_to_sqlite(
                db_path=sqlite_path,
                created_at=created_at,
                dataset_name=dataset_name,
                spec=spec,
                accuracy=accuracy,
                average_latency=avg_latency,
                predictions=predictions,
            )

            summary["datasets"][dataset_name]["results"][spec.name] = {
                "accuracy": accuracy,
                "average_latency_seconds": avg_latency,
                "model_kind": spec.kind,
                "model_domain": spec.domain,
            }
            accuracy_for_plot[spec.name].append(accuracy)
            latency_for_plot[spec.name].append(avg_latency)
            grid_rows.append((spec.name, [row["is_correct"] for row in predictions]))

            safe_dataset = (
                dataset_name.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_")
            )
            safe_model = spec.name.lower().replace(" ", "_").replace(":", "").replace(".", "")
            predictions_path = output_dir / f"predictions_{safe_dataset}_{safe_model}.jsonl"
            with predictions_path.open("w", encoding="utf-8") as file_handle:
                for row in predictions:
                    file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        safe_dataset = dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        build_correctness_grid(
            dataset_name=dataset_name,
            matrix_rows=grid_rows,
            output_path=output_dir / f"correctness_grid_{safe_dataset}.png",
        )

    dataset_names = list(datasets.keys())
    build_bar_chart(
        dataset_names=dataset_names,
        values_by_model=accuracy_for_plot,
        y_label="Accuracy",
        title="Sentiment accuracy by dataset and model",
        output_path=output_dir / "accuracy_comparison.png",
    )
    build_bar_chart(
        dataset_names=dataset_names,
        values_by_model=latency_for_plot,
        y_label="Average latency per item (seconds)",
        title="Inference latency by dataset and model",
        output_path=output_dir / "latency_comparison.png",
    )

    summary_path = output_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)
        file_handle.write("\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--social-samples",
        type=int,
        default=0,
        help="Rows to evaluate per social dataset. 0 means all rows.",
    )
    parser.add_argument(
        "--review-samples",
        type=int,
        default=2000,
        help="Rows to evaluate from the large reviews dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/isaiahjp/repos/sentiment_analysis/results",
    )
    parser.add_argument(
        "--cache-dir",
        default="/home/isaiahjp/repos/sentiment_analysis/.cache",
    )
    parser.add_argument(
        "--sqlite-path",
        default="/home/isaiahjp/repos/sentiment_analysis/results/benchmark_results.sqlite",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["instagram", "kaggle_social", "kaggle_reviews"],
        default=["instagram", "kaggle_social", "kaggle_reviews"],
        help="Datasets to benchmark. Defaults to all domains.",
    )
    args = parser.parse_args()

    summary = run_all_domain_benchmarks(
        social_samples=args.social_samples,
        review_samples=args.review_samples,
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        sqlite_path=Path(args.sqlite_path),
        selected_datasets=args.datasets,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
