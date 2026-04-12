
## Final Project Spec: Sentiment Model vs Small LLM

### 1) Objective

Compare a top-tier non-LLM sentiment classifier against Gemma 4 e2b on hard sentiment data, focusing on accuracy,
robustness, calibration, and cost/latency.

### 2) Dataset (Hugging Face)

• Primary:  dynabench/dynasent  (adversarially collected; harder than standard review datasets).
• Optional stress-test transfer:  cardiffnlp/tweet_eval  ( sentiment ) for out-of-domain evaluation.

### 3) Models

• Non-LLM baseline (top-tier):  siebert/sentiment-roberta-large-english
• LLM: Gemma 4 e2b (prompted for strict label output: negative / neutral / positive)

### 4) Experimental Design

• Standardize labels to 3-class sentiment.
• Evaluate:
  1. RoBERTa zero-shot inference (or fine-tune on DynaSent train, then test).
  2. Gemma zero-shot, then few-shot (k=3, k=8) prompting.
• Add controlled slices: negation, intensifiers, mixed sentiment, sarcasm-like examples.

### 5) Metrics

• Macro-F1 (primary), accuracy, per-class F1, confusion matrices.
• Calibration (ECE / Brier from confidence scores where available).
• Efficiency: tokens/sec, latency/sample, and cost per 1k predictions.

### 6) Deliverables

• Reproducible notebook/script pipeline, results table, error analysis with 20–30 failure cases, and a short conclusion
on when small LLMs beat or lose to specialized classifiers.
