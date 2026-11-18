import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # P4 — real reader, embedding cache, reader gate (GPU)

        All logic lives in the package; each button runs one `python -m mhtable...` command.
        Run the buttons **in order**. Every step writes to local disk first; uploads to
        HuggingFace happen only where a box says so.

        1. **Setup** (automatic): clone the repo, install vLLM + the package, run the unit tests,
           print the GPU and library versions.
        2. **Smoke test**: oracle selector + real reader on 5 dev questions (strict tier) —
           checks the vLLM install and the prompt end to end (~5 min incl. model download).
        3. **Embedding cache**: bge-m3 over every text the featuriser can look up for
           Spider+MMQA (~214k texts, a few minutes); **3b** the same for BIRD (~1–2M texts,
           ~15–30 min); optional push to `mhtable-cache`.
        4. **P4 gate**: oracle selector + real reader on N dev questions, **both tiers**;
           prints EM, answer coverage, and reader throughput (calls/hour).
        5. **Push the gate run** to `mhtable-runs` (optional).

        Needs `HF_TOKEN` (env var or the box below) — read access for the data, **write**
        access for the pushes. The reader model (Qwen2.5-7B-Instruct) is public.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import os

    REPO_URL = "https://github.com/advay333/Multi-Hop-Project.git"
    GIT_REF = "main"
    WORKDIR = os.path.expanduser("~/mhtable")
    RUNS = os.path.expanduser("~/mhtable_runs")
    EMB = os.path.expanduser("~/mhtable_emb")

    token_box = mo.ui.text(kind="password", label="HF token (leave empty if HF_TOKEN is set)", full_width=True)
    token_box
    return EMB, GIT_REF, REPO_URL, RUNS, WORKDIR, os, token_box


@app.cell
def _(GIT_REF, REPO_URL, WORKDIR, os, token_box):
    import shutil
    import subprocess
    import sys

    def run(cmd, env=None):
        """Run a command, streaming its output into the cell; raise on failure."""
        print("$", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            print(line, end="", flush=True)
        if proc.wait() != 0:
            raise RuntimeError(f"command failed with exit code {proc.returncode}")

    if token_box.value:
        os.environ["HF_TOKEN"] = token_box.value
    assert os.environ.get("HF_TOKEN"), "set HF_TOKEN or paste a token above"

    if os.path.isdir(os.path.join(WORKDIR, ".git")):
        run(["git", "-C", WORKDIR, "fetch", "--depth", "1", "origin", GIT_REF])
        run(["git", "-C", WORKDIR, "checkout", "-q", "FETCH_HEAD"])
    else:
        run(["git", "clone", "--depth", "1", "--branch", GIT_REF, REPO_URL, WORKDIR])
    # vLLM first: it pins its own torch/transformers; the package then installs on top of them.
    # A failed vLLM install is not fatal: choose backend "hf" below and continue.
    try:
        run([sys.executable, "-m", "pip", "install", "-q", "vllm"])
    except RuntimeError:
        print("\n*** vLLM did not install — set 'Reader backend' to hf below (same prompt, slower) ***\n")
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{WORKDIR}[dev,train,p4]"])
    if os.path.join(WORKDIR, "src") not in sys.path:
        sys.path.insert(0, os.path.join(WORKDIR, "src"))

    run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
    run([sys.executable, "-c", "import torch, transformers, sentence_transformers as st; "
         "print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0)); "
         "print('transformers', transformers.__version__, '| sentence-transformers', st.__version__)\n"
         "try:\n    import vllm; print('vllm', vllm.__version__)\nexcept Exception as e:\n    print('vllm NOT available:', e)"])
    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 2**30
    print(f"free disk: {free_gb:.1f} GiB | CPUs: {os.cpu_count()}")
    run([sys.executable, "-m", "pytest", "-q", WORKDIR], env=os.environ.copy())
    return run, sys


@app.cell
def _(mo):
    revision = mo.ui.text(value="data-v3", label="Data revision")
    model = mo.ui.text(value="Qwen/Qwen2.5-7B-Instruct", label="Reader model", full_width=True)
    dtype = mo.ui.dropdown(options=["float16", "bfloat16"], value="float16",
                           label="Reader precision (keep float16: same as Kaggle T4, PLAN §4b)")
    backend = mo.ui.dropdown(options=["vllm", "hf"], value="vllm", label="Reader backend (hf = fallback if vLLM fails)")
    n_gate = mo.ui.number(start=10, stop=2000, step=10, value=200, label="Gate questions per tier")
    push_emb = mo.ui.checkbox(value=True, label="Push the embedding cache to mhtable-cache")
    mo.vstack([revision, model, dtype, backend, n_gate, push_emb])
    return backend, dtype, model, n_gate, push_emb, revision


@app.cell
def _(RUNS, WORKDIR, backend, dtype, model, os, revision):
    def eval_cmd(sys_exe, run_id, limit, tiers):
        return [sys_exe, "-m", "mhtable.evaluate", "--config", os.path.join(WORKDIR, "configs/experiments/eval.yaml"),
                "--selector", "oracle", "--split", "dev", "--limit", str(limit), "--tiers", *tiers,
                "--out", RUNS, "--run-id", run_id,
                f"data.revision={revision.value}", f"reader.name={backend.value}", f"reader.model={model.value}",
                f"reader.dtype={dtype.value}"]
    return (eval_cmd,)


@app.cell
def _(mo):
    smoke = mo.ui.run_button(label="2. Smoke test (5 questions, strict)")
    smoke
    return (smoke,)


@app.cell
def _(eval_cmd, mo, os, run, smoke, sys):
    mo.stop(not smoke.value, mo.md("Press **Smoke test** to start."))
    run(eval_cmd(sys.executable, "p4-smoke", 5, ["strict"]), env=os.environ.copy())
    return


@app.cell
def _(mo):
    embed = mo.ui.run_button(label="3. Build the bge-m3 embedding cache (spider_mmqa)")
    embed
    return (embed,)


@app.cell
def _(EMB, embed, mo, os, push_emb, revision, run, sys):
    mo.stop(not embed.value, mo.md("Press **Build the embedding cache** to start."))
    run([sys.executable, "-m", "mhtable.encoders.build_cache", "--family", "spider_mmqa", "--revision", revision.value,
         "--model", "BAAI/bge-m3", "--out", EMB] + (["--push"] if push_emb.value else []), env=os.environ.copy())
    return


@app.cell
def _(mo):
    embed_bird = mo.ui.run_button(label="3b. Build the bge-m3 embedding cache (bird)")
    embed_bird
    return (embed_bird,)


@app.cell
def _(EMB, embed_bird, mo, os, push_emb, revision, run, sys):
    mo.stop(not embed_bird.value, mo.md("Press **Build the embedding cache (bird)** to start. Reads only BIRD's "
                                        "questions, schemas and index (~1–2M texts), not the 32 GB of databases."))
    run([sys.executable, "-m", "mhtable.encoders.build_cache", "--family", "bird", "--revision", revision.value,
         "--model", "BAAI/bge-m3", "--out", EMB] + (["--push"] if push_emb.value else []), env=os.environ.copy())
    return


@app.cell
def _(mo):
    gate = mo.ui.run_button(label="4. Run the P4 gate (oracle + reader, both tiers)")
    gate
    return (gate,)


@app.cell
def _(RUNS, eval_cmd, gate, mo, n_gate, os, run, sys):
    mo.stop(not gate.value, mo.md("Press **Run the P4 gate** to start."))
    gate_id = f"p4-gate-oracle-dev{int(n_gate.value)}"
    run(eval_cmd(sys.executable, gate_id, int(n_gate.value), ["strict", "relaxed"]), env=os.environ.copy())
    print(open(os.path.join(RUNS, gate_id, "result.json"), encoding="utf-8").read())
    return (gate_id,)


@app.cell
def _(mo):
    push_runs = mo.ui.run_button(label="5. Push the gate run to mhtable-runs")
    push_runs
    return (push_runs,)


@app.cell
def _(RUNS, gate_id, mo, os, push_runs, run, sys):
    mo.stop(not push_runs.value, mo.md("Press **Push the gate run** to upload it."))
    run([sys.executable, "-m", "mhtable.storage", "push", "runs", os.path.join(RUNS, gate_id), f"runs/{gate_id}",
         "-m", f"P4 gate run {gate_id}"], env=os.environ.copy())
    return


if __name__ == "__main__":
    app.run()
