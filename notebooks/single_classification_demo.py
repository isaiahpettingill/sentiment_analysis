import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    from transformers import pipeline

    from pipelines.domain_benchmarks import MODEL_SPECS, normalize_label, parse_llm_label

    return Llama, MODEL_SPECS, hf_hub_download, mo, normalize_label, parse_llm_label, pipeline


@app.cell
def _(MODEL_SPECS, mo):
    model_names = [spec.name for spec in MODEL_SPECS]
    model_choice = mo.ui.dropdown(options=model_names, value=model_names[0], label="Model")
    text_input = mo.ui.text_area(
        value="I love how easy this workflow is, but inference speed still matters.",
        label="Text",
    )
    run = mo.ui.run_button(label="Classify")
    mo.vstack([model_choice, text_input, run])
    return model_choice, run, text_input


@app.cell
def _(Llama, MODEL_SPECS, hf_hub_download, model_choice, normalize_label, parse_llm_label, pipeline, run, text_input, mo):
    if not run.value:
        mo.md("Click **Classify** to run one model on one text.")
        return

    spec = next(spec for spec in MODEL_SPECS if spec.name == model_choice.value)
    text = text_input.value

    if spec.kind == "transformer":
        classifier = pipeline("sentiment-analysis", model=spec.model)
        output = classifier(text, truncation=True)[0]
        label = normalize_label(output.get("label", ""))
        latency_note = "Latency available in benchmark runs."
        raw_output = output
    else:
        model_path = hf_hub_download(repo_id=spec.repo or "", filename=spec.file or "")
        llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)
        prompt = (
            "Classify the sentiment of this text as POSITIVE, NEGATIVE, or NEUTRAL. "
            "Reply with exactly one label: POSITIVE, NEGATIVE, or NEUTRAL.\n\n"
            f"Text: {text}\n"
            "Label:"
        )
        output = llm.create_completion(prompt=prompt, max_tokens=6, temperature=0.0)
        raw_text = output["choices"][0]["text"].strip()
        label = parse_llm_label(raw_text)
        latency_note = "Latency available in benchmark runs."
        raw_output = raw_text

    mo.md(
        "\n".join(
            [
                f"**Model:** `{spec.name}`",
                f"**Predicted label:** `{label}`",
                f"**Note:** {latency_note}",
                f"**Raw output:** `{raw_output}`",
            ]
        )
    )
    return


if __name__ == "__main__":
    app.run()
