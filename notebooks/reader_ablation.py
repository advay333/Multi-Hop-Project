import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Reader ablation — prompt p1 vs p2, Qwen2.5-7B vs 14B (GPU)

        Same setup as the P4 gate (oracle selector = gold sub-tables, the same seeded 200 dev
        questions per tier, data-v3, bf16, vLLM), so differences come from the reader alone.
        Each arm is a fresh run (`--fresh`: no cached answers), so its timing is a real
        throughput measurement. Rollouts are batched 128 questions at a time, the size of one RL
        iteration (`questions_per_iter`).

        | Arm | Model | Prompt |
        |---|---|---|
        | 7b-p1 | Qwen2.5-7B-Instruct | p1 (the gate prompt — re-run for timing) |
        | 7b-p2 | Qwen2.5-7B-Instruct | p2 (exact columns, every matching row, compare all rows) |
        | 14b-p1 | Qwen2.5-14B-Instruct | p1 |
        | 14b-p2 | Qwen2.5-14B-Instruct | p2 |

        Each arm loads its model in its own process (14B: ~30 GB download the first time). The
        last cells compare the arms and push them to `mhtable-runs`.
        Needs `HF_TOKEN` (read for data, write for the push).
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

    token_box = mo.ui.text(kind="password", label="HF token (leave empty if HF_TOKEN is set)", full_width=True)
    token_box
    return GIT_REF, REPO_URL, RUNS, WORKDIR, os, token_box


@app.cell
def _(GIT_REF, REPO_URL, WORKDIR, os, token_box):
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
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # greedy decoding; see readers/vllm.py

    if os.path.isdir(os.path.join(WORKDIR, ".git")):
        run(["git", "-C", WORKDIR, "fetch", "--depth", "1", "origin", GIT_REF])
        run(["git", "-C", WORKDIR, "checkout", "-q", "FETCH_HEAD"])
    else:
        run(["git", "clone", "--depth", "1", "--branch", GIT_REF, REPO_URL, WORKDIR])
    run([sys.executable, "-m", "pip", "install", "-q", "vllm"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{WORKDIR}[dev,train,p4]"])
    if os.path.join(WORKDIR, "src") not in sys.path:
        sys.path.insert(0, os.path.join(WORKDIR, "src"))
    run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])
    run([sys.executable, "-c", "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"])
    run([sys.executable, "-m", "pytest", "-q", WORKDIR], env=os.environ.copy())
    return run, sys


@app.cell
def _(mo):
    ARMS = {
        "7b-p1": ("Qwen/Qwen2.5-7B-Instruct", "p1"),
        "7b-p2": ("Qwen/Qwen2.5-7B-Instruct", "p2"),
        "14b-p1": ("Qwen/Qwen2.5-14B-Instruct", "p1"),
        "14b-p2": ("Qwen/Qwen2.5-14B-Instruct", "p2"),
    }
    arms = mo.ui.multiselect(options=list(ARMS), value=list(ARMS), label="Arms to run (in order)")
    n_q = mo.ui.number(start=10, stop=2000, step=10, value=200, label="Dev questions per tier (200 = same as the gate)")
    revision = mo.ui.text(value="data-v3", label="Data revision")
    go = mo.ui.run_button(label="Run the selected arms")
    mo.vstack([arms, n_q, revision, go])
    return ARMS, arms, go, n_q, revision


@app.cell
def _(ARMS, RUNS, WORKDIR, arms, go, mo, n_q, os, revision, run, sys):
    mo.stop(not go.value, mo.md("Press **Run the selected arms** to start."))
    for arm in arms.value:
        model, prompt = ARMS[arm]
        run([sys.executable, "-m", "mhtable.evaluate", "--config", os.path.join(WORKDIR, "configs/experiments/eval.yaml"),
             "--selector", "oracle", "--split", "dev", "--limit", str(int(n_q.value)), "--tiers", "strict", "relaxed",
             "--chunk", "128", "--fresh", "--out", RUNS, "--run-id", f"reader-{arm}-dev{int(n_q.value)}",
             f"data.revision={revision.value}", "reader.name=vllm", f"reader.model={model}",
             "reader.dtype=bfloat16", f"reader.prompt={prompt}"], env=os.environ.copy())
    done_arms = list(arms.value)
    return (done_arms,)


@app.cell
def _(ARMS, RUNS, done_arms, mo, n_q, os):
    import json

    rows = []
    for arm in done_arms:
        path = os.path.join(RUNS, f"reader-{arm}-dev{int(n_q.value)}", "result.json")
        if not os.path.exists(path):
            continue
        r = json.load(open(path, encoding="utf-8"))
        b = r["reader_stats"].get("backend", {})
        for tier, t in r["tiers"].items():
            rows.append({"arm": arm, "tier": tier, "em": t["em"], "set_f1": t["set_f1"],
                         "em_when_shown": t["em_when_answer_shown"], "em_on_path": t["em_on_path"],
                         "List": t["by_category"].get("List", {}).get("em"),
                         "Select": t["by_category"].get("Select", {}).get("em"),
                         "Count": t["by_category"].get("Count", {}).get("em"),
                         "Numerical": t["by_category"].get("Numerical", {}).get("em")})
        rows.append({"arm": arm, "tier": "timing", "load_s": r.get("reader_load_seconds"),
                     "calls": r["reader_stats"]["calls"], "reader_s": r["reader_stats"]["seconds"],
                     "calls_per_hour": r["reader_stats"]["calls_per_hour"],
                     "prompt_tok_per_s": b.get("prompt_tokens_per_second"),
                     "answer_tok_per_s": b.get("answer_tokens_per_second"),
                     "mean_answer_tokens": b.get("mean_answer_tokens"), "answers_cut": b.get("answers_cut"),
                     "total_min": r.get("minutes"), "model": ARMS[arm][0]})
    print(json.dumps(rows, indent=1))
    mo.ui.table(rows, selection=None)
    return


@app.cell
def _(mo):
    push = mo.ui.run_button(label="Push the arm runs to mhtable-runs")
    push
    return (push,)


@app.cell
def _(RUNS, done_arms, mo, n_q, os, push, run, sys):
    mo.stop(not push.value, mo.md("Press **Push** to upload the runs."))
    for arm in done_arms:
        rid = f"reader-{arm}-dev{int(n_q.value)}"
        run([sys.executable, "-m", "mhtable.storage", "push", "runs", os.path.join(RUNS, rid), f"runs/{rid}",
             "-m", f"reader ablation {rid}"], env=os.environ.copy())
    return


if __name__ == "__main__":
    app.run()
