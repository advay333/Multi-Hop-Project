import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full", app_title="Multi-hop table RL: inference demo")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Learning multi-hop table reasoning with RL: inference demo

        A small learned **policy** picks filters and joins over a relational database, SQLite runs
        them, and a frozen **LLM reader** answers the question from the resulting sub-table.
        Pick a question and one or more selectors to watch each one decide, step by step.

        **Run this in app view (code hidden).** On molab the first question takes about 3 minutes
        (data, embeddings and the 7B reader load once); after that each one takes a few seconds.
        **Replay mode** plays back traces saved earlier and needs no GPU.

        All logic lives in `mhtable/demo.py`; this notebook only lays it out. New runs on HF
        `mhtable-runs/prajna/` appear in the selector list automatically, and the reader can be
        swapped below.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import os

    REPO_URL = "https://github.com/advay333/Multi-Hop-Project.git"
    GIT_REF = "main"
    WORKDIR = os.path.expanduser("~/mhtable")

    mode = mo.ui.radio(options=["Live", "Replay"], value="Live", label="Mode", inline=True)
    token_box = mo.ui.text(kind="password", label="HF token (leave empty if HF_TOKEN is set)", full_width=True)
    setup_go = mo.ui.run_button(label="1. Set up")
    mo.vstack([mode, token_box, setup_go])
    return GIT_REF, REPO_URL, WORKDIR, mode, os, setup_go, token_box


@app.cell
def _(GIT_REF, REPO_URL, WORKDIR, mo, os, setup_go, token_box):
    import subprocess
    import sys

    mo.stop(not setup_go.value, mo.md("Press **1. Set up**: it clones the repo and installs it (about 1 minute)."))

    def _run(cmd):
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{out.stdout[-3000:]}")

    if token_box.value:
        os.environ["HF_TOKEN"] = token_box.value
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("set HF_TOKEN or paste a token above (the data and runs repos are private)")
    if os.path.isdir(os.path.join(WORKDIR, ".git")):
        _run(["git", "-C", WORKDIR, "fetch", "--depth", "1", "origin", GIT_REF])
        _run(["git", "-C", WORKDIR, "checkout", "-q", "FETCH_HEAD"])
    else:
        _run(["git", "clone", "--depth", "1", "--branch", GIT_REF, REPO_URL, WORKDIR])
    _run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{WORKDIR}[train,p4]"])
    if os.path.join(WORKDIR, "src") not in sys.path:
        sys.path.insert(0, os.path.join(WORKDIR, "src"))
    import mhtable.demo as D

    mo.md(f"Set up done: `{WORKDIR}` at `{GIT_REF}`.")
    return (D,)


@app.cell
def _(D, WORKDIR, mo, mode, os):
    mo.stop(mode.value != "Live")
    from mhtable.config import load_config

    # environments, embeddings, policies and the reader all load lazily, on first use
    demo = D.Demo(load_config(os.path.join(WORKDIR, "configs", "experiments", "eval.yaml")))
    return (demo,)


@app.cell
def _(mo):
    # The reader is swappable: any Hugging Face chat model works with the `hf` backend; `vllm` is
    # faster for batches but needs the vLLM wheel. The policies were trained with Qwen2.5-7B, whose
    # tokenizer keeps counting the 4,096-token budget whatever reader answers (mhtable/demo.py).
    READERS = {
        "Qwen2.5-7B-Instruct (paper setting)": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
        "Qwen2.5-3B-Instruct (small GPU)": "Qwen/Qwen2.5-3B-Instruct",
        "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "Custom (type the id)": "",
    }
    reader_pick = mo.ui.dropdown(options=READERS, value="Qwen2.5-7B-Instruct (paper setting)", label="Reader LLM")
    reader_custom = mo.ui.text(label="Custom model id", placeholder="org/model-name")
    backend = mo.ui.dropdown(options=["hf", "vllm"], value="hf", label="Backend")
    dtype = mo.ui.dropdown(options=["bfloat16", "float16"], value="bfloat16", label="dtype")
    mo.hstack([reader_pick, reader_custom, backend, dtype], justify="start")
    return backend, dtype, reader_custom, reader_pick


@app.cell
def _(backend, demo, dtype, mo, reader_custom, reader_pick):
    model_id = reader_pick.value or reader_custom.value.strip() or "Qwen/Qwen2.5-7B-Instruct"
    # records the choice and frees the previous model; the new one loads on the next question
    demo.set_reader({"name": backend.value, "model": model_id, "dtype": dtype.value})
    reader_label = f"{model_id} ({backend.value}, {dtype.value})"
    mo.md(f"Reader: **{reader_label}**. Budget counted with `{demo.cfg.env.tokenizer}`.")
    return model_id, reader_label


@app.cell
def _(D, demo, mo):
    with mo.status.spinner("Listing trained runs on HF ..."):
        specs = D.baseline_specs() + D.discover_policies()
    spec_by_name = {s.name: s for s in specs}
    split = mo.ui.dropdown(options=["dev", "test"], value="dev", label="Split")
    tier = mo.ui.radio(options=["strict", "relaxed"], value="strict", label="Action tier", inline=True)
    mo.hstack([split, tier, mo.md(f"{len(specs) - 4} trained policies found")], justify="start")
    return spec_by_name, specs, split, tier


@app.cell
def _(demo, mo, split, tier):
    with mo.status.spinner(f"Loading {split.value} questions ({tier.value}) ..."):
        items = demo.questions(split.value, tier.value)
    source = mo.ui.dropdown(options=["all", "spider", "mmqa"], value="all", label="Source")
    n_tables = mo.ui.dropdown(options=["all", "2", "3"], value="all", label="Gold tables")
    search = mo.ui.text(label="Search the question text", placeholder="e.g. singer")
    mo.hstack([source, n_tables, search], justify="start")
    return items, n_tables, search, source


@app.cell
def _(items, mo, n_tables, search, source):
    _q = search.value.lower().strip()
    shown_items = [(i, g) for i, g in items
                   if (source.value == "all" or i.source == source.value)
                   and (n_tables.value == "all" or i.n_gold_tables == int(n_tables.value))
                   and (not _q or _q in i.question.lower())][:300]
    question_pick = mo.ui.dropdown(
        options={f"{k + 1}. [{i.source} · {i.db_id} · {i.n_gold_tables} tables] {i.question[:110]}": k
                 for k, (i, _g) in enumerate(shown_items)},
        label=f"Question ({len(shown_items)} shown, first 300)", full_width=True)
    typed = mo.ui.text_area(label="...or type your own question about that question's database (it has no gold answer)",
                            full_width=True)
    mo.vstack([question_pick, typed])
    return question_pick, shown_items, typed


@app.cell
def _(mo, specs, tier):
    _ok = {f"{s.label}  [{s.group}]": s for s in specs if s.tier in (None, tier.value)}
    _default = [k for k, s in _ok.items()
                if s.name == "truncate" or s.group == "bc"
                or (s.group == "grid" and s.name.split("-")[:2] in (["ppo", "r1"], ["grpo", "r1"]) and "lr1e-5" in s.name)]
    selector_pick = mo.ui.multiselect(options={k: s.name for k, s in _ok.items()}, value=_default,
                                      label="Selectors to compare", full_width=True)
    top_k = mo.ui.slider(2, 8, value=5, label="Candidates shown per step")
    go = mo.ui.run_button(label="2. Run")
    mo.vstack([selector_pick, top_k, go])
    return go, selector_pick, top_k


@app.cell
def _(D, demo, go, mo, question_pick, reader_label, selector_pick, shown_items, spec_by_name, tier, top_k, typed):
    mo.stop(not go.value, mo.md("Choose a question and selectors, then press **2. Run**."))
    mo.stop(question_pick.value is None, mo.md("Pick a question first (a typed question uses its database)."))
    _inst, _gold = shown_items[question_pick.value]
    if typed.value.strip():
        _inst, _gold = D.Demo.typed_instance(_inst, typed.value), None
    live_traces, _errors = [], []
    with mo.status.spinner(f"Running with {reader_label} (the first run also loads the models) ...") as _sp:
        for _name in selector_pick.value:
            _spec = spec_by_name[_name]
            if _spec.uses_gold and _gold is None:
                _errors.append(f"{_spec.label}: skipped (it follows the gold path; a typed question has none)")
                continue
            _sp.update(f"{_spec.label} ...")
            try:
                live_traces.append(demo.run(_spec, _inst, _gold, tier=tier.value, top_k=top_k.value))
            except Exception as _e:  # show the error next to the others instead of stopping the demo
                _errors.append(f"{_spec.label}: {type(_e).__name__}: {_e}")
    mo.md("\n".join(f"- ⚠️ {e}" for e in _errors)) if _errors else None
    return (live_traces,)


@app.cell
def _(mo):
    import html as _html

    def _bar(p: float, chosen: bool) -> str:
        colour = "#2563eb" if chosen else "#94a3b8"
        return (f"<span style='display:inline-block;background:#e5e7eb;width:160px;height:10px;border-radius:5px'>"
                f"<span style='display:block;background:{colour};width:{max(1, int(160 * p))}px;height:10px;"
                f"border-radius:5px'></span></span> {p:.0%}")

    def render_trace(tr):
        """One selector's run: verdict, steps with probabilities, SQL, the sub-table, the answer."""
        verdict = "no gold (typed question)" if tr.em is None else ("✅ correct" if tr.em == 1.0 else "❌ wrong")
        used = min(1.0, tr.tokens / max(1, tr.budget))
        head = mo.md(
            f"### {tr.label} · {verdict}\n"
            f"**Answer:** `{tr.answer or '(empty: nothing selected)'}`"
            + (f" · **Gold:** `{tr.gold_answer}`" if tr.gold_answer is not None else "")
            + f"\n\n{tr.rows_shown} of {tr.rows_total} rows shown" + (" (cut to fit)" if tr.overflow else "")
            + f" · {len(tr.steps)} steps · {tr.seconds:.1f}s · reader `{tr.reader}`"
            + ("\n\n*The oracle follows the gold path: it is the ceiling, not a result.*" if tr.uses_gold else ""))
        budget = mo.Html(f"Tokens: {_bar(used, True)} &nbsp;({tr.tokens} of {tr.budget})")
        parts = [head, budget]
        if tr.steps:
            rows = []
            for st in tr.steps:
                cands = "".join(
                    f"<div style='margin:3px 0'>{'<b>' if c.chosen else ''}{_html.escape(c.text)}"
                    f"{'</b>' if c.chosen else ''} <span style='color:#64748b'>({c.rows} rows)</span><br>"
                    f"{_bar(c.prob, c.chosen)}</div>" for c in st.candidates)
                rows.append(f"<tr><td style='vertical-align:top;padding:4px'>{st.t + 1}</td>"
                            f"<td style='vertical-align:top;padding:4px'><b>{_html.escape(st.text)}</b><br>"
                            f"<span style='color:#64748b'>{st.rows_after} rows after · {st.n_candidates} options</span></td>"
                            f"<td style='padding:4px'>{cands or '<i>no probabilities (not a learned policy)</i>'}</td></tr>")
            parts.append(mo.Html("<table style='width:100%;border-collapse:collapse'><tr><th>#</th>"
                                 "<th style='text-align:left'>Chosen action</th>"
                                 "<th style='text-align:left'>Top candidates (policy probability)</th></tr>"
                                 + "".join(rows) + "</table>"))
            sqls = [f"-- step {st.t + 1}: {st.text}\nSELECT * {st.sql_after}" for st in tr.steps if st.sql_after]
            parts.append(mo.accordion({"SQL after each step": mo.md("```sql\n" + "\n\n".join(sqls) + "\n```")}))
        else:
            parts.append(mo.md("*No steps: this baseline sends whole tables, cut to the token budget.*"))
        sub = tr.subtable if len(tr.subtable) < 6000 else tr.subtable[:6000] + "\n... (cut for display)"
        parts.append(mo.accordion({f"Sub-table the reader saw ({len(tr.columns)} columns)": mo.md(f"```\n{sub}\n```")}))
        return mo.vstack(parts)

    def render_all(traces):
        if not traces:
            return mo.md("*No traces yet.*")
        q = traces[0]
        top = mo.md(f"## {q.question}\n`{q.source}` · database `{q.db_id}` · tier **{q.tier}**")
        table = mo.ui.table([{"selector": t.label, "answer": t.answer, "correct": None if t.em is None else bool(t.em),
                              "tokens": t.tokens, "steps": len(t.steps), "seconds": t.seconds} for t in traces],
                            selection=None)
        return mo.vstack([top, table, mo.ui.tabs({t.label: render_trace(t) for t in traces})])

    return (render_all,)


@app.cell
def _(live_traces, render_all):
    render_all(live_traces)
    return


@app.cell
def _(live_traces, mo, reader_label):
    import json as _json

    _blob = _json.dumps({"meta": {"reader": reader_label}, "traces": [t.to_dict() for t in live_traces]},
                        indent=1, default=str).encode()
    mo.download(data=_blob, filename="demo_traces.json",
                label="Save these traces for replay mode") if live_traces else None
    return


@app.cell
def _(D, mo, mode):
    mo.stop(mode.value != "Replay")
    replay_file = mo.ui.file(filetypes=[".json"], multiple=True, label="Upload saved traces (demo_traces.json)")
    replay_file
    return (replay_file,)


@app.cell
def _(D, mo, replay_file):
    import tempfile

    mo.stop(not replay_file.value, mo.md("Upload one or more saved trace files."))
    replay_groups = {}
    for _f in replay_file.value:
        with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as _tmp:
            _tmp.write(_f.contents)
        _traces, _meta = D.load_traces(_tmp.name)
        for _t in _traces:
            replay_groups.setdefault(f"{_t.question[:90]} [{_t.tier}]", []).append(_t)
    replay_pick = mo.ui.dropdown(options=list(replay_groups), value=list(replay_groups)[0],
                                 label="Saved question", full_width=True)
    replay_pick
    return replay_groups, replay_pick


@app.cell
def _(render_all, replay_groups, replay_pick):
    render_all(replay_groups[replay_pick.value])
    return


@app.cell
def _(D, mo):
    results_go = mo.ui.run_button(label="Show the results table (every run's test score on HF)")
    results_go
    return (results_go,)


@app.cell
def _(D, mo, results_go):
    mo.stop(not results_go.value)
    with mo.status.spinner("Fetching every t-*-test/result.json ..."):
        _rows = D.results_table()
    mo.ui.table([{**r, "em": None if r["em"] is None else round(100 * r["em"], 1),
                  "tokens": None if r["tokens"] is None else round(r["tokens"])} for r in _rows],
                selection=None, label="Test EM (%) per run and tier")
    return


if __name__ == "__main__":
    app.run()
