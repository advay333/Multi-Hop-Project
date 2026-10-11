"""
All project data lives in private HuggingFace repos; local disk is only a cache (DESIGN.md §4).

    raw    dataset  unmodified sources (Spider zip, MMQA mirror JSONs, BIRD zips)
    data   dataset  built artefacts: instances, gold, splits, SQLite DBs, indexes
    cache  dataset  embedding shards, reader-answer shards
    runs   model    checkpoints, configs, logs, results

Environment variables:
    HF_TOKEN               access token (read by huggingface_hub; needed for private repos)
    MHTABLE_HF_NAMESPACE   user/org owning the repos (default: advay333)
    MHTABLE_CACHE_DIR      where fetched files are cached (default: the HF cache)
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DEFAULT_NAMESPACE = "advay333"

REPOS = {
    "raw": ("mhtable-raw", "dataset"),
    "data": ("mhtable-data", "dataset"),
    "cache": ("mhtable-cache", "dataset"),
    "runs": ("mhtable-runs", "model"),
}


def namespace() -> str:
    return os.environ.get("MHTABLE_HF_NAMESPACE", DEFAULT_NAMESPACE)


def repo_id(repo: str) -> str:
    return f"{namespace()}/{_spec(repo)[0]}"


def repo_type(repo: str) -> str:
    return _spec(repo)[1]


def _spec(repo: str) -> tuple[str, str]:
    try:
        return REPOS[repo]
    except KeyError:
        raise KeyError(f"unknown repo {repo!r}; expected one of {sorted(REPOS)}") from None


def _cache_dir() -> str | None:
    return os.environ.get("MHTABLE_CACHE_DIR")


def api() -> HfApi:
    return HfApi()


def ensure_repos() -> dict[str, str]:
    """Create all four repos as private if missing. Returns {repo: url}."""
    out = {}
    for repo in REPOS:
        url = api().create_repo(repo_id(repo), repo_type=repo_type(repo), private=True, exist_ok=True)
        out[repo] = str(url)
    return out


def fetch(repo: str, path: str, revision: str | None = None) -> Path:
    """Download one file (or every file matching a glob) and return its local path.

    For a glob, returns the local snapshot root; matched files sit under it at their repo paths.
    """
    if any(ch in path for ch in "*?["):
        root = snapshot_download(repo_id(repo), repo_type=repo_type(repo), revision=revision,
                                 allow_patterns=[path], cache_dir=_cache_dir())
        return Path(root)
    return Path(hf_hub_download(repo_id(repo), path, repo_type=repo_type(repo), revision=revision,
                                cache_dir=_cache_dir()))


def push(repo: str, local_path: str | Path, path_in_repo: str, message: str) -> str:
    """Upload a file, or a folder's contents under `path_in_repo` ("" = repo root). Returns the commit sha."""
    local_path = Path(local_path)
    kw = dict(repo_id=repo_id(repo), repo_type=repo_type(repo), commit_message=message)
    if local_path.is_dir():
        info = api().upload_folder(folder_path=str(local_path), path_in_repo=path_in_repo or None, **kw)
    else:
        info = api().upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo, **kw)
    return info.oid


def push_large(repo: str, folder: str | Path) -> None:
    """Resumable upload of a large folder, mirroring its layout at the repo root.

    Uses many small commits and can be re-run after an interruption (molab sessions die);
    files already uploaded are skipped.
    """
    api().upload_large_folder(repo_id=repo_id(repo), folder_path=str(folder), repo_type=repo_type(repo))


def remote_sha256(repo: str, paths: list[str], revision: str | None = None) -> dict[str, str | None]:
    """SHA-256 of large (LFS/xet) files already in the repo, keyed by path; None if absent/small."""
    found = {p: None for p in paths}
    for info in api().get_paths_info(repo_id(repo), paths, repo_type=repo_type(repo), revision=revision):
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            found[info.path] = lfs.sha256
    return found


def tag(repo: str, name: str, message: str = "") -> None:
    api().create_tag(repo_id(repo), tag=name, repo_type=repo_type(repo), tag_message=message)


def db_path(db_ref: str, revision: str | None = None) -> Path:
    """`spider/concert_singer` -> local path of that SQLite file, fetched on first use."""
    return fetch("data", f"db/{db_ref}.sqlite", revision=revision)
