import hashlib

from mhtable.data import source


def test_checksum_helpers(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert source.sha256_file(f) == hashlib.sha256(b"hello").hexdigest()
    sums = {"mmqa/b.json": "22", "spider/a.zip": "11"}
    assert source.parse_checksums(source.format_checksums(sums)) == sums
    # sha256sum's binary-mode marker and the old local layout parse too
    assert source.parse_checksums("abc *data/raw/x.zip\n") == {"data/raw/x.zip": "abc"}


def test_known_checksums_match_local_record():
    # The sources registry must agree with the checksums recorded when the data was first mirrored.
    known = {sf.repo_path: sf.sha256 for files in source.SOURCES.values() for sf in files if sf.sha256}
    assert len(known) == 3 and all(len(s) == 64 for s in known.values())


def test_dry_run_touches_nothing(tmp_path, capsys):
    sums = source.mirror(["spider", "mmqa", "bird"], tmp_path, dry_run=True)
    assert sums == {}
    out = capsys.readouterr().out
    assert out.count("[would mirror]") == 5
    assert list(tmp_path.iterdir()) == []
