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


def _resolve_created_at(conn: sqlite3.Connection, created_at: str | None) -> str:
    if created_at:
        row = conn.execute("SELECT 1 FROM runs WHERE created_at = ? LIMIT 1", (created_at,)).fetchone()
        if row is None:
            raise ValueError(f"No run found for created_at={created_at}")
        return created_at

    row = conn.execute("SELECT MAX(created_at) FROM runs").fetchone()
    if row is None or row[0] is None:
        raise ValueError("No benchmark runs found in SQLite database")
    return str(row[0])


def _get_models(conn: sqlite3.Connection, created_at: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT model_name FROM runs WHERE created_at = ? ORDER BY model_name",
        (created_at,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _get_datasets(conn: sqlite3.Connection, created_at: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT dataset_name FROM runs WHERE created_at = ? ORDER BY dataset_name",
        (created_at,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _dataset_accuracy(conn: sqlite3.Connection, created_at: str, dataset_name: str) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT model_name, accuracy
        FROM runs
        WHERE created_at = ? AND dataset_name = ?
        """,
        (created_at, dataset_name),
    ).fetchall()
    return {str(model): float(acc) for model, acc in rows}


def _dataset_latency(conn: sqlite3.Connection, created_at: str, dataset_name: str) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT model_name, average_latency_seconds
        FROM runs
        WHERE created_at = ? AND dataset_name = ?
        """,
        (created_at, dataset_name),
    ).fetchall()
    return {str(model): float(lat) for model, lat in rows}


def _combined_accuracy(conn: sqlite3.Connection, created_at: str) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT model_name, AVG(CAST(is_correct AS REAL))
        FROM predictions
        WHERE run_id IN (SELECT run_id FROM runs WHERE created_at = ?)
        GROUP BY model_name
        """,
        (created_at,),
    ).fetchall()
    return {str(model): float(acc) for model, acc in rows}


def _combined_latency(conn: sqlite3.Connection, created_at: str) -> dict[str, float]:
    rows = conn.execute(
        """
        SELECT model_name, AVG(latency_seconds)
        FROM predictions
        WHERE run_id IN (SELECT run_id FROM runs WHERE created_at = ?)
        GROUP BY model_name
        """,
        (created_at,),
    ).fetchall()
    return {str(model): float(lat) for model, lat in rows}


def _dataset_grid_values(
    conn: sqlite3.Connection,
    created_at: str,
    dataset_name: str,
    models: list[str],
) -> list[tuple[str, list[int]]]:
    rows = conn.execute(
        """
        SELECT model_name, sample_index, is_correct
        FROM predictions
        WHERE run_id IN (
            SELECT run_id FROM runs
            WHERE created_at = ? AND dataset_name = ?
        )
        ORDER BY model_name, sample_index
        """,
        (created_at, dataset_name),
    ).fetchall()

    by_model: dict[str, list[int]] = {model: [] for model in models}
    for model_name, _, is_correct in rows:
        by_model[str(model_name)].append(int(is_correct))

    return [(model, by_model.get(model, [])) for model in models]


def _combined_grid_values(conn: sqlite3.Connection, created_at: str, models: list[str]) -> list[tuple[str, list[int]]]:
    sample_rows = conn.execute(
        """
        SELECT DISTINCT dataset_name, sample_index
        FROM predictions
        WHERE run_id IN (SELECT run_id FROM runs WHERE created_at = ?)
        ORDER BY dataset_name, sample_index
        """,
        (created_at,),
    ).fetchall()
    sample_keys = [(str(dataset_name), int(sample_index)) for dataset_name, sample_index in sample_rows]

    pred_rows = conn.execute(
        """
        SELECT dataset_name, sample_index, model_name, is_correct
        FROM predictions
        WHERE run_id IN (SELECT run_id FROM runs WHERE created_at = ?)
        """,
        (created_at,),
    ).fetchall()

    index_map: dict[tuple[str, str, int], int] = {}
    for dataset_name, sample_index, model_name, is_correct in pred_rows:
        index_map[(str(dataset_name), str(model_name), int(sample_index))] = int(is_correct)

    matrix: list[tuple[str, list[int]]] = []
    for model in models:
        values = [index_map.get((dataset_name, model, sample_index), 0) for dataset_name, sample_index in sample_keys]
        matrix.append((model, values))
    return matrix


def _plot_bar(values: dict[str, float], title: str, y_label: str, output_path: Path) -> None:
    models = list(values.keys())
    y_values = [values[model] for model in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(models, y_values)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_grid(matrix_rows: list[tuple[str, list[int]]], title: str, output_path: Path) -> None:
    if not matrix_rows:
        return
    max_len = max(len(values) for _, values in matrix_rows)
    fig_height = max(3, 0.8 * len(matrix_rows))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    for y, (model_name, values) in enumerate(matrix_rows):
        for x, value in enumerate(values):
            ax.scatter(x, y, marker="s", s=16, color="#2ca02c" if value == 1 else "#d62728")

    ax.set_title(title)
    ax.set_xlabel("Prompt index")
    ax.set_yticks(list(range(len(matrix_rows))))
    ax.set_yticklabels([model_name for model_name, _ in matrix_rows])
    ax.set_xticks(list(range(0, max_len, max(1, math.ceil(max_len / 20)))))
    ax.grid(alpha=0.15)
    ax.legend(
        handles=[Patch(color="#2ca02c", label="Correct (1)"), Patch(color="#d62728", label="Wrong (0)")],
        loc="upper right",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_report_charts(sqlite_path: Path, output_dir: Path, created_at: str | None) -> None:
    with sqlite3.connect(sqlite_path) as conn:
        run_created_at = _resolve_created_at(conn, created_at)
        models = _get_models(conn, run_created_at)
        datasets = _get_datasets(conn, run_created_at)

        for dataset_name in datasets:
            slug = _slugify(dataset_name)
            _plot_bar(
                values=_dataset_accuracy(conn, run_created_at, dataset_name),
                title=f"Accuracy by model: {dataset_name}",
                y_label="Accuracy",
                output_path=output_dir / f"report_accuracy_{slug}.png",
            )
            _plot_bar(
                values=_dataset_latency(conn, run_created_at, dataset_name),
                title=f"Compute cost (avg latency) by model: {dataset_name}",
                y_label="Average latency per item (seconds)",
                output_path=output_dir / f"report_compute_cost_{slug}.png",
            )
            _plot_grid(
                matrix_rows=_dataset_grid_values(conn, run_created_at, dataset_name, models),
                title=f"Per-prompt correctness grid: {dataset_name}",
                output_path=output_dir / f"report_grid_{slug}.png",
            )

        _plot_bar(
            values=_combined_accuracy(conn, run_created_at),
            title="Accuracy by model: all datasets combined",
            y_label="Accuracy",
            output_path=output_dir / "report_accuracy_all_datasets.png",
        )
        _plot_bar(
            values=_combined_latency(conn, run_created_at),
            title="Compute cost (avg latency) by model: all datasets combined",
            y_label="Average latency per item (seconds)",
            output_path=output_dir / "report_compute_cost_all_datasets.png",
        )
        _plot_grid(
            matrix_rows=_combined_grid_values(conn, run_created_at, models),
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
        help="Specific run timestamp (created_at) from runs table. Defaults to latest run.",
    )
    args = parser.parse_args()

    generate_report_charts(
        sqlite_path=Path(args.sqlite_path),
        output_dir=Path(args.output_dir),
        created_at=args.created_at,
    )


if __name__ == "__main__":
    main()
