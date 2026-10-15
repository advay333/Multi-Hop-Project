import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Build the environment index (phase P3)

        Runs `python -m mhtable.env.index` for one family against a data-repo revision and pushes
        `index/<db_ref>.json` files (column stats, candidate joins, cell values) to `mhtable-data`
        (all logic lives in the package, not here).

        - **spider_mmqa** — 865 databases, ~600 MB download, ~1 minute, ~30 MB output.
        - **bird** — 80 databases, ~32 GB download; slower (large tables).

        Needs an HF token with **write** access (`HF_TOKEN` env var or the box below). No GPU needed.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    import os

    REPO_URL = "https://github.com/advay333/Multi-Hop-Project.git"
    GIT_REF = "main"
    WORKDIR = os.path.expanduser("~/mhtable")

    token_box = mo.ui.text(kind="password", label="HF token (leave empty if HF_TOKEN is set)", full_width=True)
    token_box
    return GIT_REF, REPO_URL, WORKDIR, os, token_box


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
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{WORKDIR}[dev]"])
    # An editable install is only picked up by new processes; make this kernel see it too.
    if os.path.join(WORKDIR, "src") not in sys.path:
        sys.path.insert(0, os.path.join(WORKDIR, "src"))

    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 2**30
    print(f"free disk: {free_gb:.1f} GiB | CPUs: {os.cpu_count()}")
    run([sys.executable, "-m", "pytest", "-q", WORKDIR], env=os.environ.copy())
    return free_gb, run, sys


@app.cell
def _(mo):
    family = mo.ui.dropdown(options=["spider_mmqa", "bird"], value="spider_mmqa", label="Family")
    revision = mo.ui.text(value="data-v2", label="Data repo revision to index")
    go = mo.ui.run_button(label="Build index and push")
    mo.vstack([family, revision, go])
    return family, go, revision


@app.cell
def _(family, go, mo, os, revision, run, sys):
    mo.stop(not go.value, mo.md("Press **Build index and push** to start."))
    out = os.path.join(os.path.expanduser("~"), "mhtable_index", family.value)
    run([sys.executable, "-m", "mhtable.env.index", "--family", family.value, "--revision", revision.value,
         "--out", out, "--workers", str(max(1, (os.cpu_count() or 2) - 1)), "--push"], env=os.environ.copy())
    return


if __name__ == "__main__":
    app.run()
