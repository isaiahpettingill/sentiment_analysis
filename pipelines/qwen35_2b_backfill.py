import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from pipelines.domain_benchmarks import (
        ModelSpec,
        _ensure_db,
        _load_selected_datasets,
        evaluate_llm,
        save_results_to_sqlite,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from pipelines.domain_benchmarks import (
        ModelSpec,
        _ensure_db,
        _load_selected_datasets,
        evaluate_llm,
        save_results_to_sqlite,
    )


QWEN35_2B_Q4 = ModelSpec(
    name="Qwen3.5 2B Q4",
    kind="llm",
    repo="unsloth/Qwen3.5-2B-GGUF",
    file="Qwen3.5-2B-Q4_0.gguf",
    domain="general",
)


def run_qwen35_2b_backfill(
    social_samples: int,
    review_samples: int,
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
    created_at = datetime.now(UTC).isoformat()
    summary: dict[str, dict] = {"created_at": created_at, "model": QWEN35_2B_Q4.name, "datasets": {}}

    for dataset_name, rows in datasets.items():
        print(f"[dataset] {dataset_name}: {len(rows)} samples")
        progress_label = f"{dataset_name} | {QWEN35_2B_Q4.name}"
        accuracy, avg_latency, predictions = evaluate_llm(
            repo=QWEN35_2B_Q4.repo or "",
            filename=QWEN35_2B_Q4.file or "",
            rows=rows,
            progress_label=progress_label,
        )

        run_id = save_results_to_sqlite(
            db_path=sqlite_path,
            created_at=created_at,
            dataset_name=dataset_name,
            spec=QWEN35_2B_Q4,
            accuracy=accuracy,
            average_latency=avg_latency,
            predictions=predictions,
        )
        summary["datasets"][dataset_name] = {
            "run_id": run_id,
            "samples": len(rows),
            "accuracy": accuracy,
            "average_latency_seconds": avg_latency,
        }
        print(
            f"[done] {dataset_name} | {QWEN35_2B_Q4.name} | "
            f"accuracy={accuracy:.4f} latency={avg_latency:.4f}s"
        )

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

    summary = run_qwen35_2b_backfill(
        social_samples=args.social_samples,
        review_samples=args.review_samples,
        cache_dir=Path(args.cache_dir),
        sqlite_path=Path(args.sqlite_path),
        selected_datasets=args.datasets,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
