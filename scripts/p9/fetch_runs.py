"""Download Prajna run results (no checkpoints, no reader caches) from HF mhtable-runs into runs/hf/.

    python scripts/p9/fetch_runs.py -cov0.5-s0 -cov1-s0        # any substrings of run names
    python scripts/p9/fetch_runs.py                            # everything under prajna/

Run from the project root. `mhtable-runs` is a private HF *model* repo (needs HF_TOKEN / login).
"""
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

REPO = "advay333/mhtable-runs"
keys = sys.argv[1:]
files = [f for f in HfApi().list_repo_files(REPO)
         if f.startswith("prajna/") and not f.endswith(".pt") and "reader_cache" not in f
         and (not keys or any(k in f for k in keys))]
print(len(files), "files")
for f in files:
    p = hf_hub_download(REPO, f, local_dir="runs/hf")
    print(f, os.path.getsize(p))
