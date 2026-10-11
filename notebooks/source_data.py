import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Mirror raw data to HuggingFace (phase P0)

        1. Clones and installs the project from GitHub.
        2. Runs the test suite and a HuggingFace connectivity check (the P0 gate).
        3. Mirrors Spider, MMQA and BIRD into the private `mhtable-raw` repo with
           `python -m mhtable.data.source` (all logic lives there, not here).

        Needs an HF token with **write** access. Set the `HF_TOKEN` environment variable, or
        paste it in the box below. No GPU needed.
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
    print(f"free disk: {free_gb:.1f} GiB (BIRD train.zip needs ~9 GiB)")
    return free_gb, run, sys


@app.cell
def _(WORKDIR, os, run, sys):
    # P0 gate: tests pass here, and the private repos are reachable with this token.
    run([sys.executable, "-m", "pytest", "-q", WORKDIR, "-m", "not slow"], env=os.environ.copy())
    run([sys.executable, "-c",
         "from mhtable import storage; print(storage.ensure_repos())"], env=os.environ.copy())
    return


@app.cell
def _(mo):
    sources = mo.ui.multiselect(options=["spider", "mmqa", "bird"], value=["spider", "mmqa", "bird"],
                                label="Sources to mirror")
    go = mo.ui.run_button(label="Mirror to HuggingFace")
    mo.vstack([sources, go])
    return go, sources


@app.cell
def _(free_gb, go, mo, os, run, sources, sys):
    mo.stop(not go.value, mo.md("Press **Mirror to HuggingFace** to start."))
    if "bird" in sources.value and free_gb < 12:
        print(f"WARNING: only {free_gb:.1f} GiB free; BIRD train.zip (8.9 GB) may not fit. "
              "Mirror spider/mmqa first, or run BIRD from a machine with more disk.")
    run([sys.executable, "-m", "mhtable.data.source", "--sources", *sources.value], env=os.environ.copy())
    return


if __name__ == "__main__":
    app.run()
