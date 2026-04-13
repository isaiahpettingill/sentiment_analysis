default:
    @just --list

sync:
    uv sync

pipeline-domain-benchmarks:
    uv run python pipelines/domain_benchmarks.py

pipeline-social-instagram:
    uv run python pipelines/domain_benchmarks.py --social-samples 56 --review-samples 0

pipeline-social-kaggle:
    uv run python pipelines/domain_benchmarks.py --social-samples 499 --review-samples 0

pipeline-reviews-kaggle:
    uv run python pipelines/domain_benchmarks.py --social-samples 0 --review-samples 2000

pipeline-all-domains:
    uv run python pipelines/domain_benchmarks.py

charts-report:
    uv run python pipelines/report_charts.py

report-all:
    uv run python pipelines/domain_benchmarks.py
    uv run python pipelines/report_charts.py

notebook-imdb:
    uv run marimo edit notebooks/imdb_pipeline_notebook.py

notebook-twitter:
    uv run marimo edit notebooks/twitter_pipeline_notebook.py

notebook-single:
    uv run marimo edit notebooks/single_classification_demo.py

notebooks-all:
    uv run marimo edit notebooks/imdb_pipeline_notebook.py
    uv run marimo edit notebooks/twitter_pipeline_notebook.py
    uv run marimo edit notebooks/single_classification_demo.py
