import pytest

from mhtable import storage


def test_repo_ids_follow_namespace(monkeypatch):
    monkeypatch.setenv("MHTABLE_HF_NAMESPACE", "someone")
    assert storage.repo_id("data") == "someone/mhtable-data"
    assert storage.repo_type("runs") == "model"
    with pytest.raises(KeyError):
        storage.repo_id("nope")


def test_push_dispatches_file_vs_folder(monkeypatch, tmp_path):
    calls = []

    class FakeApi:
        def upload_file(self, **kw):
            calls.append(("file", kw["path_in_repo"]))
            return type("I", (), {"oid": "f1"})()

        def upload_folder(self, **kw):
            calls.append(("folder", kw["path_in_repo"]))
            return type("I", (), {"oid": "d1"})()

    monkeypatch.setattr(storage, "api", lambda: FakeApi())
    (tmp_path / "x.txt").write_text("x")
    assert storage.push("raw", tmp_path / "x.txt", "a/x.txt", "m") == "f1"
    assert storage.push("raw", tmp_path, "", "m") == "d1"
    assert calls == [("file", "a/x.txt"), ("folder", None)]


@pytest.mark.network
def test_repos_reachable():
    # Run explicitly with: pytest -m network   (needs HF_TOKEN)
    assert set(storage.ensure_repos()) == set(storage.REPOS)


def test_push_large_without_upload_large_folder_skips_files_already_uploaded(tmp_path, monkeypatch):
    import hashlib

    from mhtable import storage

    (tmp_path / "db" / "bird").mkdir(parents=True)
    a, b = tmp_path / "db" / "bird" / "a.sqlite", tmp_path / "db" / "bird" / "b.sqlite"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "junk").write_bytes(b"x")

    class Api:  # huggingface_hub 2.x: no upload_large_folder
        pass

    pushed = []
    monkeypatch.setattr(storage, "api", lambda: Api())
    monkeypatch.setattr(storage, "remote_sha256",
                        lambda repo, paths, revision=None: {"db/bird/a.sqlite": hashlib.sha256(b"aaa").hexdigest()})
    monkeypatch.setattr(storage, "push", lambda repo, p, rel, msg: pushed.append(rel))
    storage.push_large("data", tmp_path)
    assert pushed == ["db/bird/b.sqlite"]
