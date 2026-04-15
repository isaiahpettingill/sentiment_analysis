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

pipeline-gemma4-e4b:
    uv run python pipelines/domain_benchmarks.py --dataset instagram --model "Gemma 4 E4B Q4"
    uv run python pipelines/domain_benchmarks.py --dataset kaggle_social --model "Gemma 4 E4B Q4"
    uv run python pipelines/domain_benchmarks.py --dataset kaggle_reviews --model "Gemma 4 E4B Q4"
    uv run python pipelines/report_charts.py

pipeline-all: pipeline-instagram pipeline-kaggle-social pipeline-kaggle-reviews charts

charts:
    uv run python pipelines/report_charts.py