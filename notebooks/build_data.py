import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Build the data repo (phase P1)

        Runs `python -m mhtable.data.build` for one family and pushes the output to the private
        `mhtable-data` repo (all logic lives in the package, not here).

        - **spider_mmqa** — Spider + MMQA, split together. ~1.2 GB download, ~600 MB output, minutes.
        - **bird** — BIRD train + dev. Needs **~55 GB free disk** (8.9 GB zip, nested zip, ~33 GB of
          databases); run it only if the disk check below allows.

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
    tag = mo.ui.text(value="", label="Tag after push (e.g. data-v1; leave empty to skip)")
    go = mo.ui.run_button(label="Build and push")
    mo.vstack([family, tag, go])
    return family, go, tag


@app.cell
def _(WORKDIR, family, free_gb, go, mo, os, run, sys, tag):
    mo.stop(not go.value, mo.md("Press **Build and push** to start."))
    if family.value == "bird" and free_gb < 55:
        raise RuntimeError(f"only {free_gb:.0f} GiB free; the BIRD build needs ~55 GiB")
    out = os.path.join(os.path.expanduser("~"), "mhtable_build", family.value)
    cmd = [sys.executable, "-m", "mhtable.data.build", "--family", family.value, "--out", out, "--push"]
    if tag.value.strip():
        cmd += ["--tag", tag.value.strip()]
    run(cmd, env=os.environ.copy())
    return


if __name__ == "__main__":
    app.run()
