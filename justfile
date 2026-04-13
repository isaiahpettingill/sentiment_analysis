default:
    @just --list

sync:
    uv sync

setup-cuda-remote:
    uv sync
    uv pip uninstall llama-cpp-python torch torchvision torchaudio
    uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
    CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --no-cache-dir --force-reinstall llama-cpp-python
    uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

pipeline-domain-benchmarks:
    uv run python pipelines/domain_benchmarks.py

pipeline-social-instagram:
    uv run python pipelines/domain_benchmarks.py --social-samples 56 --review-samples 0

pipeline-social-kaggle:
    uv run python pipelines/domain_benchmarks.py --social-samples 499 --review-samples 0

pipeline-reviews-kaggle:
    uv run python pipelines/domain_benchmarks.py --social-samples 0 --review-samples 500

pipeline-all-domains:
    uv run python pipelines/domain_benchmarks.py

pipeline-qwen35-2b-backfill:
    uv run python pipelines/qwen35_2b_backfill.py

charts-report:
    uv run python pipelines/report_charts.py

report-all:
    uv run python pipelines/domain_benchmarks.py
    uv run python pipelines/report_charts.py

report-qwen35-2b-backfill:
    uv run python pipelines/qwen35_2b_backfill.py
    uv run python pipelines/report_charts.py

report-rerun-accurate:
    uv sync
    uv run python pipelines/domain_benchmarks.py --datasets kaggle_reviews --review-samples 500
    # uv run python pipelines/qwen35_2b_backfill.py
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
