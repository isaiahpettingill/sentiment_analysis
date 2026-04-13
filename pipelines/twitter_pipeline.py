import argparse
import json

from datasets import load_dataset
from transformers import pipeline


LABEL_MAP = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE",
}


def run_twitter_pipeline(
    split: str,
    limit: int,
    model_name: str,
    output_path: str,
) -> None:
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    sentiment = pipeline("sentiment-analysis", model=model_name)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            pred = sentiment(row["text"], truncation=True)[0]
            record = {
                "dataset": "cardiffnlp/tweet_eval:sentiment",
                "split": split,
                "index": i,
                "text": row["text"],
                "gold_label": row.get("label"),
                "gold_label_name": LABEL_MAP.get(row.get("label"), "UNKNOWN"),
                "prediction": {
                    "label": pred["label"],
                    "score": pred["score"],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--model",
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )
    parser.add_argument(
        "--output",
        default="/home/isaiahjp/repos/sentiment_analysis/twitter_pipeline_output.jsonl",
    )
    args = parser.parse_args()

    run_twitter_pipeline(args.split, args.limit, args.model, args.output)
