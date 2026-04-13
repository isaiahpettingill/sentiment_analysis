import argparse
import json
import re

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import pipeline


def _normalize_label(label: str) -> str:
    label = label.strip().upper()
    if "POS" in label:
        return "POSITIVE"
    if "NEG" in label:
        return "NEGATIVE"
    return label


def _detect_text_column(sample: dict) -> str:
    for key, value in sample.items():
        if isinstance(value, str) and value.strip():
            return key
    raise ValueError("No non-empty text column found in dataset row")


def _build_llm(repo_id: str, filename: str, verbose: bool = False) -> Llama:
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=verbose,
    )


def _llm_sentiment_label(llm: Llama, text: str) -> tuple[str, str]:
    prompt = (
        "Classify the sentiment of the text as POSITIVE or NEGATIVE. "
        "Reply with exactly one word: POSITIVE or NEGATIVE.\n\n"
        f"Text: {text}\n"
        "Label:"
    )

    response = llm.create_completion(
        prompt=prompt,
        max_tokens=4,
        temperature=0.0,
    )
    raw = response["choices"][0]["text"].strip()

    match = re.search(r"\b(POSITIVE|NEGATIVE)\b", raw.upper())
    if match:
        return match.group(1), raw

    fallback = re.search(r"\b(pos|neg)\w*\b", raw.lower())
    if fallback:
        return _normalize_label(fallback.group(0)), raw

    return "UNPARSEABLE", raw


def _compare_row(
    row: dict,
    dataset_name: str,
    split: str,
    index: int,
    text_key: str,
    classifier_model: str,
    classifier,
    llm_repo: str,
    llm_file: str,
    llm: Llama,
) -> dict:
    text = row[text_key]

    clf_output = classifier(text, truncation=True)[0]
    clf_label = _normalize_label(clf_output["label"])

    llm_label, llm_raw = _llm_sentiment_label(llm, text)

    return {
        "dataset": dataset_name,
        "split": split,
        "index": index,
        "text_column": text_key,
        "text": text,
        "classifier": {
            "model": classifier_model,
            "label": clf_label,
            "score": clf_output["score"],
            "raw": clf_output,
        },
        "llm": {
            "repo": llm_repo,
            "file": llm_file,
            "label": llm_label,
            "raw_output": llm_raw,
        },
        "agreement": clf_label == llm_label,
    }


def compare_single_sample(
    dataset_name: str,
    split: str,
    index: int,
    classifier_model: str,
    llm_repo: str,
    llm_file: str,
    text_column: str | None,
) -> dict:
    ds = load_dataset(dataset_name, split=split)
    row = ds[index]

    text_key = text_column or _detect_text_column(row)
    classifier = pipeline("sentiment-analysis", model=classifier_model)
    llm = _build_llm(llm_repo, llm_file, verbose=args.verbose)

    return _compare_row(
        row=row,
        dataset_name=dataset_name,
        split=split,
        index=index,
        text_key=text_key,
        classifier_model=classifier_model,
        classifier=classifier,
        llm_repo=llm_repo,
        llm_file=llm_file,
        llm=llm,
    )


def compare_full_split(
    dataset_name: str,
    split: str,
    classifier_model: str,
    llm_repo: str,
    llm_file: str,
    text_column: str | None,
    output_path: str,
    verbose: bool = False,
) -> None:
    ds = load_dataset(dataset_name, split=split)
    first_row = ds[0]
    text_key = text_column or _detect_text_column(first_row)

    classifier = pipeline("sentiment-analysis", model=classifier_model)
    llm = _build_llm(llm_repo, llm_file, verbose=verbose)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            result = _compare_row(
                row=row,
                dataset_name=dataset_name,
                split=split,
                index=idx,
                text_key=text_key,
                classifier_model=classifier_model,
                classifier=classifier,
                llm_repo=llm_repo,
                llm_file=llm_file,
                llm=llm,
            )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imdb")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--classifier-model",
        default="Elron/deberta-v3-large-sentiment",
    )
    parser.add_argument(
        "--llm-repo",
        default="unsloth/gemma-4-E2B-it-GGUF",
    )
    parser.add_argument(
        "--llm-file",
        default="gemma-4-E2B-it-Q4_0.gguf",
    )
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.all:
        if not args.output:
            raise ValueError("--output is required when using --all")
        compare_full_split(
            dataset_name=args.dataset,
            split=args.split,
            classifier_model=args.classifier_model,
            llm_repo=args.llm_repo,
            llm_file=args.llm_file,
            text_column=args.text_column,
            output_path=args.output,
            verbose=args.verbose,
        )
        return

    result = compare_single_sample(
        dataset_name=args.dataset,
        split=args.split,
        index=args.index,
        classifier_model=args.classifier_model,
        llm_repo=args.llm_repo,
        llm_file=args.llm_file,
        text_column=args.text_column,
    )

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
