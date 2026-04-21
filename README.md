# Sentiment Analysis: Traditional Classifier vs Small LLM

Compare a top-tier non-LLM sentiment classifier against Gemma 4 2e on hard sentiment data.

## Objective

Evaluate accuracy, robustness, calibration, and cost/latency trade-offs between:
- **Non-LLM baseline**: siebert/sentiment-roberta-large-english
- **LLM**: Gemma 4 2e (prompted for 3-class output: negative / neutral / positive)

## Dataset

- **Primary**: [dynabench/dynasent](https://huggingface.co/datasets/dynabench/dynasent) — adversarially collected, harder than standard review datasets
- **Stress-test**: [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (sentiment) for out-of-domain evaluation

## Setup

```bash
uv sync
```

## Run

```bash
uv run python main.py
```

## Metrics

- Macro-F1 (primary), accuracy, per-class F1, confusion matrices
- Calibration: ECE / Brier scores
- Efficiency: tokens/sec, latency/sample, cost per 1k predictions

## Project Structure

```
├── pyproject.toml          # Python dependencies
├── main.py                 # Entry point
├── report_tex/             # LaTeX report
└── SPEC.md                 # Full experiment specification
```
