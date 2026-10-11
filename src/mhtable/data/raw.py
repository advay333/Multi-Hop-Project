"""
Access to the mirrored raw sources (`raw` HF repo), including zip extraction.

Archives are extracted once into `MHTABLE_WORK_DIR` (default `~/.cache/mhtable/extract`), in a
folder keyed by the archive's checksum, so re-runs are free and a changed archive never mixes
with an old extraction. Nested archives (BIRD ships `train_databases.zip` inside `train.zip`)
are extracted on demand with `extract_nested`.

`local_dir` (dev only): read raw files from a local folder instead of HF — they are located by
file name, so the old `data/raw/` layout works. Builds meant for upload must use HF.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from mhtable import storage
from mhtable.data.source import sha256_file


def work_dir() -> Path:
    d = Path(os.environ.get("MHTABLE_WORK_DIR", Path.home() / ".cache" / "mhtable" / "extract"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class RawSource:
    def __init__(self, revision: str | None = None, local_dir: str | Path | None = None):
        self.revision = revision
        self.local_dir = Path(local_dir) if local_dir else None

    def file(self, repo_path: str) -> Path:
        if self.local_dir is None:
            return storage.fetch("raw", repo_path, revision=self.revision)
        hits = list(self.local_dir.rglob(Path(repo_path).name))
        if len(hits) != 1:
            raise FileNotFoundError(f"{repo_path}: expected one match under {self.local_dir}, found {len(hits)}")
        return hits[0]

    def extracted(self, repo_path: str) -> Path:
        """Extract a zip from the raw repo; returns the extraction folder."""
        return extract(self.file(repo_path))


def extract(archive: Path) -> Path:
    key = f"{archive.stem}-{_quick_digest(archive)}"
    dest = work_dir() / key
    marker = dest / ".complete"
    if not marker.exists():
        print(f"extracting {archive.name} -> {dest}", flush=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        marker.write_text(archive.name)
    return dest


def extract_nested(root: Path, name: str) -> Path:
    """Find archive `name` somewhere under `root` and extract it next to itself."""
    return extract(find_one(root, name))


def find_one(root: Path, name: str) -> Path:
    hits = [p for p in root.rglob(name) if "__MACOSX" not in p.parts]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one {name!r} under {root}, found {[str(h) for h in hits]}")
    return hits[0]


def _quick_digest(path: Path) -> str:
    # Hashing a 9 GB zip on every call is slow; cache the digest next to the file by size+mtime.
    st = path.stat()
    memo = work_dir() / f".digest-{path.name}-{st.st_size}-{int(st.st_mtime)}"
    if memo.exists():
        return memo.read_text()
    digest = sha256_file(path)[:12]
    memo.write_text(digest)
    return digest
