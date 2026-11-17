import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # data-v3 rebuild (PLAN.md §4a)

        One notebook for the whole rebuild, in order:

        1. build + push **spider_mmqa** (minutes)
        2. build + push **bird** (needs **~55 GB free disk**; the longest step)
        3. build + push the env **index** for both, from the folders just built (no re-download)
        4. **verify**: oracle pass rates, G-6 coverage on every question (`analysis.reach`), and a
           real-execution check on a BIRD sample (`analysis.coverage`)
        5. **tag `data-v3`**: a separate button; press it only after reading step 4

        Every step runs one `python -m mhtable...` command; all logic lives in the package.
        Needs an HF token with **write** access. No GPU needed. Finished steps are skipped, so if
        the session dies, re-run the notebook and press the button again.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import os

    REPO_URL = "https://github.com/advay333/Multi-Hop-Project.git"
    GIT_REF = "main"
    WORKDIR = os.path.expanduser("~/mhtable")
    OUT = os.path.expanduser("~/mhtable_v3")  # build/<family>/, index/<family>/ and reports live here

    token_box = mo.ui.text(kind="password", label="HF token (leave empty if HF_TOKEN is set)", full_width=True)
    token_box
    return GIT_REF, OUT, REPO_URL, WORKDIR, os, token_box


@app.cell
def _(GIT_REF, REPO_URL, WORKDIR, os, token_box):
    import shutil
    import subprocess
    import sys

    def run(cmd):
        """Run a command, streaming its output into the cell; raise on failure."""
        print("$", " ".join(cmd), flush=True)
        _p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
        for _line in _p.stdout:
            print(_line, end="", flush=True)
        if _p.wait() != 0:
            raise RuntimeError(f"command failed with exit code {_p.returncode}")

    if token_box.value:
        os.environ["HF_TOKEN"] = token_box.value
    assert os.environ.get("HF_TOKEN"), "set HF_TOKEN or paste a token above"

    if os.path.isdir(os.path.join(WORKDIR, ".git")):
        run(["git", "-C", WORKDIR, "fetch", "--depth", "1", "origin", GIT_REF])
        run(["git", "-C", WORKDIR, "checkout", "-q", "FETCH_HEAD"])
    else:
        run(["git", "clone", "--depth", "1", "--branch", GIT_REF, REPO_URL, WORKDIR])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{WORKDIR}[dev]"])
    if os.path.join(WORKDIR, "src") not in sys.path:
        sys.path.insert(0, os.path.join(WORKDIR, "src"))
    run(["git", "-C", WORKDIR, "log", "-1", "--oneline"])  # should be the commit with the BIRD fixes

    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 2**30
    WORKERS = str(max(1, (os.cpu_count() or 2) - 1))
    print(f"free disk: {free_gb:.1f} GiB | CPUs: {os.cpu_count()}")
    run([sys.executable, "-m", "pytest", "-q", WORKDIR])
    return WORKERS, free_gb, run, subprocess, sys


@app.cell
def _(mo):
    go = mo.ui.run_button(label="Run steps 1–4 (build, index, verify)")
    go
    return (go,)


@app.cell
def _(OUT, WORKERS, free_gb, go, mo, os, run, sys):
    mo.stop(not go.value, mo.md("Press **Run steps 1–4** to start. Nothing is tagged here."))

    # Steps 1–2: data build + push (no tag yet: the tag must cover data + index).
    for _fam in ("spider_mmqa", "bird"):
        if os.path.exists(os.path.join(OUT, "build", _fam, "stats", f"{_fam}.json")):
            print(f"[skip] {_fam} already built")
            continue
        if _fam == "bird" and free_gb < 55:
            raise RuntimeError(f"only {free_gb:.0f} GiB free; the BIRD build needs ~55 GiB")
        run([sys.executable, "-m", "mhtable.data.build", "--family", _fam,
             "--out", os.path.join(OUT, "build", _fam), "--workers", WORKERS, "--push"])

    # Step 3: env index from the local build folders, pushed to index/ on main.
    for _fam in ("spider_mmqa", "bird"):
        _marker = os.path.join(OUT, "index", _fam, ".pushed")
        if os.path.exists(_marker):
            print(f"[skip] {_fam} index already pushed")
            continue
        run([sys.executable, "-m", "mhtable.env.index", "--family", _fam,
             "--data-dir", os.path.join(OUT, "build", _fam), "--out", os.path.join(OUT, "index", _fam),
             "--workers", WORKERS, "--push"])
        open(_marker, "w").close()
    steps_done = True
    print("steps 1–3 done: data and index are on `main` of mhtable-data (not tagged)")
    return (steps_done,)


@app.cell
def _(OUT, mo, os, steps_done):
    mo.stop(not steps_done)
    import json

    # Step 4a: oracle pass rates and eligibility counts, straight from the build stats.
    _rows, _lines = [], []
    for _fam in ("spider_mmqa", "bird"):
        _st = json.load(open(os.path.join(OUT, "build", _fam, "stats", f"{_fam}.json")))
        _lines.append(f"- **{_fam} splits**: `{json.dumps(_st['splits'])}`")
        for _src, _s in _st["sources"].items():
            _lines.append(f"- **{_src}** execution/eligibility: `{json.dumps(_s['exec'])}`")
            for _tier in ("strict", "relaxed"):
                _o = _s["oracle"][_tier]
                _rows.append({"source": _src, "tier": _tier, "eligible": _o["eligible"], "passed": _o["passed"],
                              "pass rate": _o["pass_rate"], "failures": json.dumps(_o["fail_reasons"]),
                              "questions in tier": _s[_tier]})
    mo.vstack([
        mo.md("### 4a. Oracle (expected: Spider/MMQA ≥ 98%; BIRD relaxed ≈ 93% at most, "
              "**no `oracle_no_rowid`**, no new `oracle_error`)"),
        mo.ui.table(_rows, selection=None),
        mo.md("\n".join(_lines)),
    ])
    return (json,)


@app.cell
def _(OUT, json, mo, os, steps_done, subprocess, sys):
    mo.stop(not steps_done)
    # Step 4b: G-6 reachability on every question (candidate membership only, no SQL, fast).
    # analysis.reach reads gold/instances/index from HF `main`, where steps 1–3 just pushed them.
    _rows = []
    for _fam, _splits in (("spider_mmqa", ["train", "dev"]), ("bird", ["train", "bird_dev"])):
        _out = os.path.join(OUT, f"reach_{_fam}.json")
        if not os.path.exists(_out):
            print(f"analysis.reach {_fam} {_splits} ...", flush=True)
            _res = subprocess.run([sys.executable, "-m", "mhtable.analysis.reach", "--family", _fam,
                                   "--splits", *_splits, "--data-revision", "main", "--index-revision", "main",
                                   "--misses-out", os.path.join(OUT, f"reach_{_fam}_misses.jsonl")],
                                  capture_output=True, text=True, env=os.environ.copy())
            print(_res.stderr[-3000:], flush=True)
            if _res.returncode != 0:
                raise RuntimeError(f"analysis.reach failed for {_fam}")
            open(_out, "w").write(_res.stdout)
        for _key, _r in json.load(open(_out)).items():
            _rows.append({"family": _fam, "split/tier": _key, "questions": _r["questions"],
                          "reachability": _r["reachability"], "|A| mean": _r["action_set"]["mean"],
                          "|A| max": _r["action_set"]["max"],
                          "top sole blockers": json.dumps(dict(list(_r["sole_blocker"].items())[:4]))})
    mo.vstack([mo.md("### 4b. G-6 reachability (gate 85%; local dry run Spider+MMQA train 91.0% / 86.2%; "
                     "BIRD offline estimate train 88.6% / 79.4%)"), mo.ui.table(_rows, selection=None)])
    return


@app.cell
def _(OUT, json, mo, os, run, steps_done, sys):
    mo.stop(not steps_done)
    # Step 4c: the environment with real SQL (look-ahead counts, random-policy rollouts) on a BIRD
    # sample: the first execution of composite JOINs, WITHOUT ROWID keys and evidence anchors.
    _out = os.path.join(OUT, "coverage_bird_train_sample.json")
    if not os.path.exists(_out):
        run([sys.executable, "-m", "mhtable.analysis.coverage", "--family", "bird", "--split", "train",
             "--data-dir", os.path.join(OUT, "build", "bird"), "--index-dir", os.path.join(OUT, "index", "bird"),
             "--edge-threshold", "0.6667", "--limit", "500", "--random-episodes", "300", "--out", _out])
    _r = json.load(open(_out))
    mo.md(
        "### 4c. BIRD execution check (500 questions per tier, 300 random episodes)\n\n"
        + "\n".join(f"- **{_t}**: reachability {_r[_t].get('reachability')}, random-policy crashes "
                    f"**{_r[_t]['random_policy']['crashes']}** (must be 0), max |A| "
                    f"{_r[_t]['random_policy']['candidates_per_state']['max']}" for _t in ("strict", "relaxed"))
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 5 — tag `data-v3`

        Only after step 4 looks right (send Claude the three step-4 tables first). The tag freezes
        data + index together, so every later run can name exactly which data it used.
        """
    )
    confirm = mo.ui.checkbox(label="I checked step 4; tag the current main as data-v3")
    tag_go = mo.ui.run_button(label="Tag data-v3")
    mo.vstack([confirm, tag_go])
    return confirm, tag_go


@app.cell
def _(confirm, mo, tag_go):
    mo.stop(not (tag_go.value and confirm.value), mo.md("Not tagged."))
    from mhtable import storage

    storage.tag("data", "data-v3", "data-v3: BIRD oracle fixes, LIKE->=, outer joins, gold_relation_empty filter, new index")
    mo.md("Tagged **data-v3** on mhtable-data.")
    return


if __name__ == "__main__":
    app.run()
