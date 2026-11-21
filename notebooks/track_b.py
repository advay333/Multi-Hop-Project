import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Track B — BIRD look-ahead speed (PLAN.md §4a / §4b)

        **Problem.** Every candidate action gets a look-ahead `COUNT(*)`. On BIRD's biggest
        databases a JOIN on an unindexed column makes SQLite build a temporary index inside every
        query (seconds each; a 1M-row test took 28 s → 0.016 s with a real index).

        **Fix.** Build permanent SQLite indexes on join and filter columns of large tables
        (`mhtable.env.sqlindex`), plus code fixes already in the repo (look-ahead memo, bounded
        `materialise`). Indexes change no query result, only speed.

        Steps (each one `python -m mhtable...` command, skipped if already done — re-run after a
        crash):

        1. download BIRD (data-v3: DBs, schemas, instances, gold, splits, env index) as real files
        2. speed profile **before** (6 largest DBs, 2 min each)
        3. build the indexes in place (biggest DBs first)
        4. speed profile **after** (same DBs, same seed)
        5. the BIRD environment check that never finished: G-6 coverage + random-policy crash
           test with real SQL, **15 questions from every DB** (train and bird_dev)
        6. **push** the changed DB files to `main` of mhtable-data, then **tag `data-v4`** — two
           separate buttons; press them only after reading steps 2–5

        CPU only (no GPU). Needs **~50 GB free disk** and an HF token with **write** access
        (read is enough until step 6). Expected ≈ 1.5–3 h before the push.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import os

    REPO_URL = "https://github.com/advay333/Multi-Hop-Project.git"
    GIT_REF = "main"
    WORKDIR = os.path.expanduser("~/mhtable")
    OUT = os.path.expanduser("~/mhtable_trackb")
    DATA = os.path.join(OUT, "data")  # real-file copy of the BIRD part of mhtable-data
    REVISION = "data-v3"

    token_box = mo.ui.text(kind="password", label="HF token (leave empty if HF_TOKEN is set)", full_width=True)
    token_box
    return DATA, GIT_REF, OUT, REPO_URL, REVISION, WORKDIR, os, token_box


@app.cell
def _(GIT_REF, OUT, REPO_URL, WORKDIR, os, token_box):
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
    if os.path.join(WORKDIR, "src") not in sys.path:  # an editable install made after the kernel
        sys.path.insert(0, os.path.join(WORKDIR, "src"))  # started is not on its path (push cell imports mhtable)
    run(["git", "-C", WORKDIR, "log", "-1", "--oneline"])  # must include mhtable/env/sqlindex.py
    run([sys.executable, "-m", "pytest", "-q", os.path.join(WORKDIR, "tests", "test_trackb.py"),
         os.path.join(WORKDIR, "tests", "test_env.py")])

    os.makedirs(OUT, exist_ok=True)
    free_gb = shutil.disk_usage(OUT).free / 2**30
    WORKERS = str(max(1, (os.cpu_count() or 2) - 1))
    print(f"free disk: {free_gb:.1f} GiB | CPUs: {os.cpu_count()}")
    return WORKERS, free_gb, run, subprocess, sys


@app.cell
def _(mo):
    go = mo.ui.run_button(label="Run steps 1–5 (download, profile, index, profile, check)")
    go
    return (go,)


@app.cell
def _(DATA, OUT, REVISION, WORKERS, free_gb, go, mo, os, run, sys):
    mo.stop(not go.value, mo.md("Press **Run steps 1–5** to start. Nothing is uploaded here."))

    # Step 1: real files (not cache links), because step 3 modifies the DBs in place.
    _marker = os.path.join(OUT, ".downloaded")
    if not os.path.exists(_marker):
        if free_gb < 50:
            raise RuntimeError(f"only {free_gb:.0f} GiB free; need ~50 GiB (32 GB DBs + indexes)")
        run([sys.executable, "-m", "mhtable.storage", "download", "data", DATA, "--revision", REVISION,
             "--include", "db/bird/*", "schemas/bird.json", "instances/bird.jsonl", "gold/bird.jsonl",
             "splits/bird/*", "index/bird/*"])
        open(_marker, "w").close()
    else:
        print("[skip] already downloaded")

    _speed = [sys.executable, "-m", "mhtable.analysis.speed", "--data-dir", DATA, "--family", "bird",
              "--split", "train", "--top", "6", "--budget-s", "120", "--episodes", "200", "--seed", "0"]

    # Step 2: profile before indexing (each slow look-ahead stops at the 2 s timeout -> -1).
    SPEED_BEFORE = os.path.join(OUT, "speed_before.json")
    if not os.path.exists(SPEED_BEFORE) and not os.path.exists(os.path.join(DATA, "db_index_manifest.json")):
        run(_speed + ["--out", SPEED_BEFORE])

    # Step 3: indexes (plan first, then build). Small DBs are not touched.
    # Re-running after a crash is safe: finished indexes are skipped, the manifest lists them all.
    if not os.path.exists(os.path.join(DATA, "db_index_manifest.json")):
        _plan = os.path.join(OUT, "index_plan.json")
        run([sys.executable, "-m", "mhtable.env.sqlindex", "--data-dir", DATA, "--family", "bird", "--dry-run",
             "--plan-out", _plan])
        import json as _json
        import shutil as _shutil

        _need = _json.load(open(_plan))["est_added_bytes"] / 2**30
        _free = _shutil.disk_usage(DATA).free / 2**30
        print(f"indexes need ~{_need:.1f} GiB (estimate), free {_free:.1f} GiB", flush=True)
        if _free < 1.5 * _need + 5:  # + sort temp files of parallel workers
            raise RuntimeError("not enough disk for all indexes: free space, or re-run sqlindex with --no-anchors "
                               "(join columns only) and tell Claude")
        run([sys.executable, "-m", "mhtable.env.sqlindex", "--data-dir", DATA, "--family", "bird",
             "--workers", WORKERS])
    else:
        print("[skip] indexes already built")

    # Step 4: profile after, same DBs and seed.
    SPEED_AFTER = os.path.join(OUT, "speed_after.json")
    if not os.path.exists(SPEED_AFTER):
        run(_speed + ["--out", SPEED_AFTER])

    # Step 5: environment check with real SQL, 15 questions from every DB, 500 random episodes.
    COVERAGE = {}
    for _split in ("train", "bird_dev"):
        COVERAGE[_split] = os.path.join(OUT, f"coverage_bird_{_split}.json")
        if not os.path.exists(COVERAGE[_split]):
            run([sys.executable, "-m", "mhtable.analysis.coverage", "--family", "bird", "--split", _split,
                 "--data-dir", DATA, "--index-dir", DATA, "--edge-threshold", "0.6667",
                 "--per-db", "15", "--random-episodes", "500", "--out", COVERAGE[_split]])
    steps_done = True
    return COVERAGE, SPEED_AFTER, SPEED_BEFORE, steps_done


@app.cell
def _(COVERAGE, DATA, SPEED_AFTER, SPEED_BEFORE, mo, os, steps_done):
    mo.stop(not steps_done)
    import json

    def _load(p):
        return json.load(open(p)) if os.path.exists(p) else None

    _before, _after = _load(SPEED_BEFORE), _load(SPEED_AFTER)
    _rows = []
    for _ref, _a in (_after or {}).get("databases", {}).items():
        _b = (_before or {}).get("databases", {}).get(_ref, {})
        _bl, _al = _b.get("lookahead", {}), _a["lookahead"]
        _rows.append({"db": _ref, "GiB": _a["file_gib"],
                      "look-ahead mean s (before → after)": f"{_bl.get('mean_s')} → {_al['mean_s']}",
                      "timeout rate (before → after)": f"{_bl.get('timeout_rate')} → {_al['timeout_rate']}",
                      "episode mean s (before → after)": f"{(_b.get('episode_s') or {}).get('mean')} → "
                                                         f"{(_a['episode_s'] or {}).get('mean')}",
                      "episodes in 120 s (before → after)": f"{_b.get('episodes')} → {_a['episodes']}",
                      "materialise timeouts": _a["materialise"]["timeouts"], "crashes": _a["crashes"]})
    _man = _load(os.path.join(DATA, "db_index_manifest.json")) or {}
    _cov = []
    for _split, _p in COVERAGE.items():
        _r = _load(_p)
        for _t in ("strict", "relaxed"):
            _c = _r[_t]
            _cov.append({"split": _split, "tier": _t, "questions": _c["instances"],
                         "reachability": _c["reachability"],
                         "random-policy crashes (must be 0)": _c["random_policy"]["crashes"],
                         "materialise timeouts": _c["random_policy"].get("materialise_timeouts"),
                         "look-ahead timeout rate (gold path)": _c["lookahead"]["timeout_rate"],
                         "look-ahead timeout rate (random episodes)":
                             _c["random_policy"].get("lookahead", {}).get("timeout_rate"),
                         "look-ahead mean s": _c["lookahead"]["mean_s"],
                         "slowest DBs (s)": json.dumps(_c["slowest_dbs_s"])})
    mo.vstack([
        mo.md("### Steps 2 + 4 — speed, before vs after indexing"), mo.ui.table(_rows, selection=None),
        mo.md(f"### Step 3 — indexes: {len(_man.get('changed', []))} of {_man.get('databases')} DBs changed, "
              f"{_man.get('bytes_before', 0) / 2**30:.1f} → {_man.get('bytes_after', 0) / 2**30:.1f} GiB in "
              f"{_man.get('seconds', 0) / 60:.0f} min; changed: `{', '.join(_man.get('changed', []))}`"),
        mo.md("### Step 5 — BIRD environment check (15 questions per DB; crash-free is the gate; "
              "reachability on this sample should be near the offline G-6: train 87.4% / 79.5%, "
              "bird_dev 80.1% / 73.6%)"),
        mo.ui.table(_cov, selection=None),
        mo.md("Send Claude: this cell's tables, plus `speed_after.json` and the two coverage JSONs from "
              f"`{os.path.dirname(SPEED_AFTER)}` if anything looks off (crashes, timeouts > 1%)."),
    ])
    return (json,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Step 6 — push the indexed DBs, then tag `data-v4`

        Only after steps 2–5 look right. **Push** uploads only the changed DB files (to the same
        paths, `db/bird/<name>.sqlite`) plus the index manifest (`stats/db_index_bird.json`) to
        `main` of mhtable-data; it is resumable — press again if the session dies. Then **tag**
        `data-v4` = data-v3 + indexed BIRD databases (nothing else changes: same questions, gold,
        splits and env index). Spider/MMQA runs can keep `data-v3`.
        """
    )
    push_ok = mo.ui.checkbox(label="I checked steps 2–5; upload the changed DB files")
    push_go = mo.ui.run_button(label="Push indexed DBs")
    mo.vstack([push_ok, push_go])
    return push_go, push_ok


@app.cell
def _(DATA, OUT, json, mo, os, push_go, push_ok, run, sys):
    mo.stop(not (push_go.value and push_ok.value), mo.md("Not pushed."))
    # Stage hard links of the changed files at their repo paths (no copy), then a resumable upload.
    from mhtable import storage as _st

    _refs = _st.api().list_repo_refs(_st.repo_id("data"), repo_type="dataset")
    _main = next(b.target_commit for b in _refs.branches if b.name == "main")
    # data-v3 is an annotated tag: its ref points at a tag object, so resolve it to its commit
    _v3 = _st.api().list_repo_commits(_st.repo_id("data"), repo_type="dataset", revision="data-v3")[0].commit_id
    print(f"main {_main[:10]} | data-v3 {_v3[:10]} -> " + ("main is at data-v3: good" if _main == _v3 else
          "WARNING: main has commits after data-v3; data-v4 will include them. Tell Claude before tagging."))
    _man = json.load(open(os.path.join(DATA, "db_index_manifest.json")))
    _stage = os.path.join(OUT, "push")
    os.makedirs(os.path.join(_stage, "db", "bird"), exist_ok=True)
    os.makedirs(os.path.join(_stage, "stats"), exist_ok=True)
    for _ref in _man["changed"]:
        _src, _dst = os.path.join(DATA, "db", f"{_ref}.sqlite"), os.path.join(_stage, "db", f"{_ref}.sqlite")
        if not os.path.exists(_dst):
            os.link(_src, _dst)
    with open(os.path.join(_stage, "stats", "db_index_bird.json"), "w") as _f:
        json.dump(_man, _f, indent=2)
    run([sys.executable, "-m", "mhtable.storage", "push-large", "data", _stage])
    pushed = True
    mo.md(f"Pushed {len(_man['changed'])} DB files + manifest to `main`. Now tag.")
    return (pushed,)


@app.cell
def _(mo, pushed):
    mo.stop(not pushed)
    tag_ok = mo.ui.checkbox(label="Push finished without errors; tag the current main as data-v4")
    tag_go = mo.ui.run_button(label="Tag data-v4")
    mo.vstack([tag_ok, tag_go])
    return tag_go, tag_ok


@app.cell
def _(mo, run, sys, tag_go, tag_ok):
    mo.stop(not (tag_go.value and tag_ok.value), mo.md("Not tagged."))
    run([sys.executable, "-m", "mhtable.storage", "tag", "data", "data-v4", "-m",
         "data-v4: data-v3 + SQLite indexes on join/filter columns of large BIRD tables (env.sqlindex)"])
    mo.md("Tagged **data-v4** on mhtable-data.")
    return


if __name__ == "__main__":
    app.run()
