import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from pipelines.imdb_pipeline import run_imdb_pipeline
    return mo, run_imdb_pipeline


@app.cell
def _(mo):
    mo.md("""# IMDB pipeline notebook""")
    return


@app.cell
def _(mo):
    split = mo.ui.dropdown(options=["train", "test", "unsupervised"], value="test", label="Split")
    limit = mo.ui.number(start=1, stop=5000, value=50, label="Rows")
    model = mo.ui.text(value="Elron/deberta-v3-large-sentiment", label="Model")
    output = mo.ui.text(value="/home/isaiahjp/repos/sentiment_analysis/imdb_pipeline_output.jsonl", label="Output JSONL")
    run = mo.ui.run_button(label="Run pipeline")
    mo.vstack([split, limit, model, output, run])
    return split, limit, model, output, run


@app.cell
def _(run, run_imdb_pipeline, split, limit, model, output, mo):
    if run.value:
        run_imdb_pipeline(split.value, int(limit.value), model.value, output.value)
        mo.md(f"Wrote results to `{output.value}`")
    else:
        mo.md("Click **Run pipeline** to execute.")
    return


if __name__ == "__main__":
    app.run()
