"""
Mirror the raw sources into the `raw` HuggingFace repo, verified by SHA-256.

Run on molab (datacenter bandwidth), not the laptop:

    python -m mhtable.data.source                         # all sources
    python -m mhtable.data.source --sources spider mmqa   # a subset
    python -m mhtable.data.source --dry-run               # show what would happen

For each file: if the repo already holds it with the expected checksum, skip; otherwise
download, verify, upload, and delete the local copy (unless --keep). Files with a known
checksum must match it exactly — a mismatch means the upstream source changed and stops the
run. Files without one (BIRD) get their checksum recorded on first upload, and later runs
verify against that recording. The repo's CHECKSUMS.sha256 and SOURCES.md are rewritten at
the end.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

from mhtable import storage


@dataclass(frozen=True)
class SourceFile:
    repo_path: str  # destination path inside the raw repo
    sha256: str | None  # expected checksum; None = record on first upload
    url: str | None = None  # http(s) download
    gdrive_file: str | None = None  # Google Drive file id
    gdrive_folder: str | None = None  # Google Drive folder id (file located by name inside it)


SOURCES: dict[str, list[SourceFile]] = {
    # yale-lily.github.io/spider -> official Google Drive release (train + dev + test DBs)
    "spider": [
        SourceFile("spider/spider.zip",
                   "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b",
                   gdrive_file="1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"),
    ],
    # The paper's anonymous.4open.science link is dead; this Drive folder is the replacement.
    "mmqa": [
        SourceFile("mmqa/Synthesized_two_table.json",
                   "dbc24b22f7e491fbed9a62c72530d60c18243c0132e5097cede5393d600875ab",
                   gdrive_folder="1XQ9djKSK4yjxLWAmHMzsyPAKMqdCIpXo"),
        SourceFile("mmqa/Synthesized_three_table.json",
                   "4052303ef0721efd699f5c99e642d9db5e8082fb0d0d8b0955bfdb8566d0401b",
                   gdrive_folder="1XQ9djKSK4yjxLWAmHMzsyPAKMqdCIpXo"),
    ],
    # bird-bench.github.io official zips (dev 346 MB, train 8.9 GB)
    "bird": [
        SourceFile("bird/dev.zip", None, url="https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"),
        SourceFile("bird/train.zip", None, url="https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"),
    ],
}

CHECKSUMS_PATH = "CHECKSUMS.sha256"
SOURCES_MD_PATH = "SOURCES.md"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def parse_checksums(text: str) -> dict[str, str]:
    """`<sha>  <path>` lines (sha256sum format, optional '*') -> {path: sha}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line:
            sha, path = line.split(maxsplit=1)
            out[path.lstrip("*")] = sha
    return out


def format_checksums(sums: dict[str, str]) -> str:
    return "".join(f"{sha}  {path}\n" for path, sha in sorted(sums.items()))


def _http_download(url: str, dest: Path) -> None:
    """Resumable streaming download (the BIRD train zip is 8.9 GB)."""
    done = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:  # already complete
            return
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        total = int(r.headers.get("Content-Length", 0)) + (done if mode == "ab" else 0)
        with open(dest, mode) as f:
            got = done if mode == "ab" else 0
            for i, block in enumerate(r.iter_content(chunk_size=1 << 22)):
                f.write(block)
                got += len(block)
                if i % 256 == 0 and total:
                    print(f"    {dest.name}: {got / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)


def download(sf: SourceFile, workdir: Path) -> Path:
    import gdown

    name = Path(sf.repo_path).name
    if sf.url:
        dest = workdir / name
        _http_download(sf.url, dest)
        return dest
    if sf.gdrive_file:
        dest = workdir / name
        if not dest.exists():
            gdown.download(id=sf.gdrive_file, output=str(dest), quiet=False)
        return dest
    if sf.gdrive_folder:
        folder = workdir / f"gdrive_{sf.gdrive_folder}"
        hits = list(folder.rglob(name)) if folder.exists() else []
        if not hits:
            gdown.download_folder(id=sf.gdrive_folder, output=str(folder), quiet=False)
            hits = list(folder.rglob(name))
        if not hits:
            raise FileNotFoundError(f"{name} not found in Drive folder {sf.gdrive_folder}")
        return hits[0]
    raise ValueError(f"{sf.repo_path}: no download location configured")


def _existing_checksums() -> dict[str, str]:
    try:
        return parse_checksums(storage.fetch("raw", CHECKSUMS_PATH).read_text())
    except Exception:  # first run: file not in the repo yet
        return {}


def _sources_md(sums: dict[str, str]) -> str:
    lines = ["# Raw sources", "", f"Last mirrored: {dt.date.today().isoformat()} by `mhtable.data.source`.", "",
             "| Path | Origin | SHA-256 |", "|---|---|---|"]
    for files in SOURCES.values():
        for sf in files:
            origin = (sf.url or (f"Google Drive file {sf.gdrive_file}" if sf.gdrive_file
                                 else f"Google Drive folder {sf.gdrive_folder}"))
            lines.append(f"| `{sf.repo_path}` | {origin} | `{sums.get(sf.repo_path, 'not yet mirrored')}` |")
    return "\n".join(lines) + "\n"


def mirror(names: list[str], workdir: Path, keep: bool = False, dry_run: bool = False) -> dict[str, str]:
    if not dry_run:
        storage.ensure_repos()
    recorded = {} if dry_run else _existing_checksums()
    files = [sf for n in names for sf in SOURCES[n]]
    remote = {} if dry_run else storage.remote_sha256("raw", [sf.repo_path for sf in files])

    for sf in files:
        expected = sf.sha256 or recorded.get(sf.repo_path)
        have = remote.get(sf.repo_path)
        if have and (expected is None or have == expected):
            print(f"[skip] {sf.repo_path} already mirrored ({have[:12]})")
            recorded[sf.repo_path] = have
            continue
        if dry_run:
            print(f"[would mirror] {sf.repo_path} (expected sha {expected[:12] if expected else 'unknown'})")
            continue

        print(f"[download] {sf.repo_path}")
        local = download(sf, workdir)
        got = sha256_file(local)
        if expected and got != expected:
            raise RuntimeError(f"{sf.repo_path}: checksum mismatch (expected {expected}, got {got}). "
                               "The upstream source changed — investigate before mirroring.")
        print(f"[upload] {sf.repo_path} ({local.stat().st_size / 2**20:.0f} MiB, sha {got[:12]})")
        storage.push("raw", local, sf.repo_path, f"Mirror {sf.repo_path}")
        recorded[sf.repo_path] = got
        if not keep:
            local.unlink()

    if not dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / CHECKSUMS_PATH).write_text(format_checksums(recorded))
            (Path(tmp) / SOURCES_MD_PATH).write_text(_sources_md(recorded))
            storage.push("raw", Path(tmp), "", "Update CHECKSUMS.sha256 and SOURCES.md")
    return recorded


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    p.add_argument("--workdir", type=Path, default=None, help="download scratch dir (default: a temp dir)")
    p.add_argument("--keep", action="store_true", help="keep downloaded files after upload")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="mhtable_raw_"))
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        sums = mirror(args.sources, workdir, keep=args.keep, dry_run=args.dry_run)
    finally:
        if args.workdir is None and not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
    print(format_checksums(sums), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
