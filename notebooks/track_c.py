import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Track C — BC with bge-m3, the P6 baselines, and the BC gate (GPU)

        Closes Track C (PLAN.md §4b). Each button runs `python -m mhtable...` commands; nothing
        here contains logic.

        1. **Setup** (automatic): clone, install vLLM + the package, run the unit tests.
        2. **Behaviour cloning** with the frozen bge-m3 cache (`configs/experiments/bc_bge.yaml`),
           strict and relaxed, seed 0 — about 15–30 min each, mostly SQL look-ahead on the CPU.
        3. **Evaluation suite** on the **full dev split, both tiers**, reader Qwen2.5-7B-Instruct
           (bf16, prompt p1 — locked after the ablation): oracle ceiling, B1 truncate, B2 random
           rows, the heuristic selector, and the BC policies — about 5 min per selector.
        4. **Table** of all rows, then **push** everything to `mhtable-runs`.

        Gates read from the table: **P6** = numbers for every baseline on dev; **P7** = BC beats
        the heuristic on dev join-path accuracy (and we see how BC compares on EM).
        Needs `HF_TOKEN` (read for data + embeddings, write for the push).
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
    run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    run([sys.executable, "-m", "pytest", "-q", WORKDIR], env=os.environ.copy())
    return run, sys


@app.cell
def _(mo):
    bc_go = mo.ui.run_button(label="2. Train BC with bge-m3 (strict, then relaxed)")
    bc_go
    return (bc_go,)


@app.cell
def _(RUNS, WORKDIR, bc_go, mo, os, run, sys):
    mo.stop(not bc_go.value, mo.md("Press **Train BC** to start (skip it if both checkpoints already exist)."))
    for _tier in ("strict", "relaxed"):
        run([sys.executable, "-m", "mhtable.train", "--config", os.path.join(WORKDIR, "configs/experiments/bc_bge.yaml"),
             "--out", RUNS, "--run-id", f"bc-bge-{_tier}-s0", f"env.tier={_tier}"], env=os.environ.copy())
    return


@app.cell
def _(mo):
    SELECTORS = ["oracle", "truncate", "random_rows", "heuristic", "bc"]
    chosen = mo.ui.multiselect(options=SELECTORS, value=SELECTORS, label="Selectors to evaluate (full dev, both tiers)")
    ev_go = mo.ui.run_button(label="3. Run the evaluation suite")
    mo.vstack([chosen, ev_go])
    return SELECTORS, chosen, ev_go


@app.cell
def _(RUNS, WORKDIR, chosen, ev_go, mo, os, run, sys):
    mo.stop(not ev_go.value, mo.md("Press **Run the evaluation suite** to start."))

    def evaluate(selector, run_id, tiers, extra=()):
        run([sys.executable, "-m", "mhtable.evaluate", "--config", os.path.join(WORKDIR, "configs/experiments/eval.yaml"),
             "--selector", selector, "--split", "dev", "--tiers", *tiers, "--chunk", "128", "--out", RUNS,
             "--run-id", run_id, *extra], env=os.environ.copy())

    for _sel in chosen.value:
        if _sel == "bc":  # one checkpoint per tier
            for _tier in ("strict", "relaxed"):
                evaluate("policy", f"c-bc-{_tier}-dev", [_tier],
                         ["--policy", os.path.join(RUNS, f"bc-bge-{_tier}-s0", "policy.pt"), "--reported"])
        else:
            evaluate(_sel, f"c-{_sel}-dev", ["strict", "relaxed"], [] if _sel == "oracle" else ["--reported"])
    eval_done = list(chosen.value)
    return (eval_done,)


@app.cell
def _(RUNS, eval_done, mo, os):
    import json

    def _load(run_id):
        path = os.path.join(RUNS, run_id, "result.json")
        return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None

    table = []
    for _sel in eval_done:
        ids = [f"c-bc-{t}-dev" for t in ("strict", "relaxed")] if _sel == "bc" else [f"c-{_sel}-dev"]
        for rid in ids:
            r = _load(rid)
            if r is None:
                continue
            for tier, t in r["tiers"].items():
                table.append({"selector": _sel, "tier": tier, "n": t["n"], "em": t.get("em"), "set_f1": t.get("set_f1"),
                              "path_match": t.get("path_match"), "answer_shown": t.get("answer_in_subtable"),
                              "tokens": t.get("tokens"), "overflow": t.get("overflow"),
                              "edge_P": t.get("edge_precision"), "edge_R": t.get("edge_recall"),
                              "calls_per_hour": r["reader_stats"].get("calls_per_hour")})
    for _tier in ("strict", "relaxed"):  # BC's own training result (mock-reader dev, join-path)
        r = _load(f"bc-bge-{_tier}-s0")
        if r:
            table.append({"selector": "bc (train log)", "tier": _tier, "n": r["dev"]["n"],
                          "path_match": r["dev"]["path_match"], "em": None,
                          "tokens": r["dev"]["tokens"], "train_subset_path": r["train_subset"]["path_match"],
                          "best_epoch": r["best_epoch"], "minutes": r["minutes"]})
    print(json.dumps(table, indent=1))
    mo.ui.table(table, selection=None)
    return


@app.cell
def _(mo):
    push = mo.ui.run_button(label="4. Push the BC and evaluation runs to mhtable-runs")
    push
    return (push,)


@app.cell
def _(RUNS, mo, os, push, run, sys):
    mo.stop(not push.value, mo.md("Press **Push** to upload."))
    for _rid in sorted(os.listdir(RUNS)):
        if _rid.startswith(("bc-bge-", "c-")):
            run([sys.executable, "-m", "mhtable.storage", "push", "runs", os.path.join(RUNS, _rid), f"runs/{_rid}",
                 "-m", f"track C {_rid}"], env=os.environ.copy())
    return


if __name__ == "__main__":
    app.run()
