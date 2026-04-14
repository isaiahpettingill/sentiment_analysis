import argparse
import math
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _slugify(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(":", "")
        .replace("/", "_")
    )


def _selected_runs(conn: sqlite3.Connection, created_at: str | None) -> list[dict]:
    if created_at:
        rows = conn.execute(
            """
            SELECT run_id, created_at, dataset_name, model_name, accuracy, average_latency_seconds
            FROM runs
            WHERE created_at = ?
            ORDER BY dataset_name, model_name
            """,
            (created_at,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT r.run_id, r.created_at, r.dataset_name, r.model_name, r.accuracy, r.average_latency_seconds
            FROM runs r
            JOIN (
                SELECT dataset_name, model_name, MAX(created_at) AS max_created_at
                FROM runs
                GROUP BY dataset_name, model_name
            ) latest
            ON r.dataset_name = latest.dataset_name
            AND r.model_name = latest.model_name
            AND r.created_at = latest.max_created_at
            ORDER BY r.dataset_name, r.model_name
            """,
        ).fetchall()

    selected = [
        {
            "run_id": str(run_id),
            "created_at": str(run_created_at),
            "dataset_name": str(dataset_name),
            "model_name": str(model_name),
            "accuracy": float(accuracy),
            "average_latency_seconds": float(latency),
        }
        for run_id, run_created_at, dataset_name, model_name, accuracy, latency in rows
    ]
    if not selected:
        raise ValueError("No benchmark runs found for requested selection")
    return selected


def _group_by_dataset(selected_runs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in selected_runs:
        grouped.setdefault(row["dataset_name"], []).append(row)
    for dataset_name in grouped:
        grouped[dataset_name] = sorted(
            grouped[dataset_name], key=lambda item: item["model_name"]
        )
    return grouped


def _models(selected_runs: list[dict]) -> list[str]:
    return sorted({row["model_name"] for row in selected_runs})


def _dataset_accuracy(rows: list[dict]) -> dict[str, float]:
    return {row["model_name"]: row["accuracy"] for row in rows}


def _dataset_latency(rows: list[dict]) -> dict[str, float]:
    return {row["model_name"]: row["average_latency_seconds"] for row in rows}


def _combined_accuracy(
    conn: sqlite3.Connection, run_ids: list[str]
) -> dict[str, float]:
    placeholders = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        f"""
        SELECT model_name, AVG(CAST(is_correct AS REAL))
        FROM predictions
        WHERE run_id IN ({placeholders})
        GROUP BY model_name
        ORDER BY model_name
        """,
        run_ids,
    ).fetchall()
    return {str(model): float(acc) for model, acc in rows}


def _combined_latency(conn: sqlite3.Connection, run_ids: list[str]) -> dict[str, float]:
    placeholders = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        f"""
        SELECT model_name, AVG(latency_seconds)
        FROM predictions
        WHERE run_id IN ({placeholders})
        GROUP BY model_name
        ORDER BY model_name
        """,
        run_ids,
    ).fetchall()
    return {str(model): float(lat) for model, lat in rows}


def _dataset_grid_values(
    conn: sqlite3.Connection, rows: list[dict], models: list[str]
) -> list[tuple[str, list[int]]]:
    by_model: dict[str, list[int]] = {model: [] for model in models}
    for row in rows:
        pred_rows = conn.execute(
            """
            SELECT is_correct
            FROM predictions
            WHERE run_id = ?
            ORDER BY sample_index
            """,
            (row["run_id"],),
        ).fetchall()
        by_model[row["model_name"]] = [int(value) for (value,) in pred_rows]
    return [(model, by_model.get(model, [])) for model in models if by_model.get(model)]


def _combined_grid_values(
    conn: sqlite3.Connection, selected_runs: list[dict], models: list[str]
) -> list[tuple[str, list[int]]]:
    sample_keys: set[tuple[str, int]] = set()
    prediction_map: dict[tuple[str, str, int], int] = {}

    for row in selected_runs:
        pred_rows = conn.execute(
            """
            SELECT sample_index, is_correct
            FROM predictions
            WHERE run_id = ?
            ORDER BY sample_index
            """,
            (row["run_id"],),
        ).fetchall()
        for sample_index, is_correct in pred_rows:
            key = (row["dataset_name"], int(sample_index))
            sample_keys.add(key)
            prediction_map[
                (row["dataset_name"], row["model_name"], int(sample_index))
            ] = int(is_correct)

    ordered_samples = sorted(sample_keys, key=lambda item: (item[0], item[1]))
    matrix: list[tuple[str, list[int]]] = []
    for model in models:
        values = [
            prediction_map.get((dataset_name, model, sample_index), 0)
            for dataset_name, sample_index in ordered_samples
        ]
        if values:
            matrix.append((model, values))
    return matrix


MODEL_COLORS = {
    "Gemma 4 E2B": "#1f77b4",
    "Qwen3.5 0.8B Q4": "#ff7f0e",
    "Qwen3.5 2B Q4": "#2ca02c",
    "Specialized: BERTweet Sentiment": "#d62728",
    "Specialized: Reviews-SST2": "#9467bd",
    "Specialized: Twitter-RoBERTa": "#8c564b",
}


def _plot_bar(
    values: dict[str, float], title: str, y_label: str, output_path: Path
) -> None:
    models = list(values.keys())
    y_values = [values[model] for model in models]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x_positions = list(range(len(models)))
    bar_colors = [MODEL_COLORS.get(model, "#7f7f7f") for model in models]
    bars = ax.bar(
        x_positions,
        y_values,
        width=0.6,
        color=bar_colors,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, max(y_values) * 1.1 if max(y_values) > 0 else 1)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_grid(
    matrix_rows: list[tuple[str, list[int]]], title: str, output_path: Path
) -> None:
    if not matrix_rows:
        return
    max_len = max(len(values) for _, values in matrix_rows)
    fig_height = max(3, 0.8 * len(matrix_rows))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for y, (_, values) in enumerate(matrix_rows):
        for x, value in enumerate(values):
            ax.scatter(
                x, y, marker="s", s=16, color="#2ca02c" if value == 1 else "#d62728"
            )

    ax.set_title(title)
    ax.set_xlabel("Prompt index")
    ax.set_yticks(list(range(len(matrix_rows))))
    ax.set_yticklabels([model_name for model_name, _ in matrix_rows])
    ax.set_xticks(list(range(0, max_len, max(1, math.ceil(max_len / 20)))))
    ax.grid(alpha=0.15)
    ax.legend(
        handles=[
            Patch(color="#2ca02c", label="Correct (1)"),
            Patch(color="#d62728", label="Wrong (0)"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_report_charts(
    sqlite_path: Path, output_dir: Path, created_at: str | None
) -> None:
    with sqlite3.connect(sqlite_path) as conn:
        selected = _selected_runs(conn, created_at)
        grouped = _group_by_dataset(selected)
        models = _models(selected)
        run_ids = [row["run_id"] for row in selected]

        for dataset_name, rows in grouped.items():
            slug = _slugify(dataset_name)
            _plot_bar(
                values=_dataset_accuracy(rows),
                title=f"Accuracy by model: {dataset_name}",
                y_label="Accuracy",
                output_path=output_dir / f"report_accuracy_{slug}.png",
            )
            _plot_bar(
                values=_dataset_latency(rows),
                title=f"Compute cost (avg latency) by model: {dataset_name}",
                y_label="Average latency per item (seconds)",
                output_path=output_dir / f"report_compute_cost_{slug}.png",
            )
            _plot_grid(
                matrix_rows=_dataset_grid_values(conn, rows, models),
                title=f"Per-prompt correctness grid: {dataset_name}",
                output_path=output_dir / f"report_grid_{slug}.png",
            )

        _plot_bar(
            values=_combined_accuracy(conn, run_ids),
            title="Accuracy by model: all datasets combined",
            y_label="Accuracy",
            output_path=output_dir / "report_accuracy_all_datasets.png",
        )
        _plot_bar(
            values=_combined_latency(conn, run_ids),
            title="Compute cost (avg latency) by model: all datasets combined",
            y_label="Average latency per item (seconds)",
            output_path=output_dir / "report_compute_cost_all_datasets.png",
        )
        _plot_grid(
            matrix_rows=_combined_grid_values(conn, selected, models),
            title="Per-prompt correctness grid: all datasets combined",
            output_path=output_dir / "report_grid_all_datasets.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sqlite-path",
        default="/home/isaiahjp/repos/sentiment_analysis/results/benchmark_results.sqlite",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/isaiahjp/repos/sentiment_analysis/results/report_charts",
    )
    parser.add_argument(
        "--created-at",
        default=None,
        help="If provided, chart one benchmark batch. If omitted, uses latest run per model per dataset.",
    )
    args = parser.parse_args()

    generate_report_charts(
        sqlite_path=Path(args.sqlite_path),
        output_dir=Path(args.output_dir),
        created_at=args.created_at,
    )


if __name__ == "__main__":
    main()
