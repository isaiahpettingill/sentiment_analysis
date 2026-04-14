default:
    @just --list

sync:
    uv sync

pipeline-instagram:
    uv run python pipelines/domain_benchmarks.py --dataset instagram

pipeline-kaggle-social:
    uv run python pipelines/domain_benchmarks.py --dataset kaggle_social

pipeline-kaggle-reviews:
    uv run python pipelines/domain_benchmarks.py --dataset kaggle_reviews

pipeline-all: pipeline-instagram pipeline-kaggle-social pipeline-kaggle-reviews charts

charts:
    uv run python pipelines/report_charts.py