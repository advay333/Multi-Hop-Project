import pytest

from mhtable.config import Config, load_config


def test_base_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 0\nalgo: {name: ppo, clip: 0.2}\nreader: {name: mock}\n")
    (tmp_path / "exp.yaml").write_text("_base_: base.yaml\nalgo: {clip: 0.1}\n")
    cfg = load_config(tmp_path / "exp.yaml", ["reader.name=vllm", "algo.epochs=4", "tags=[a, b]"])
    assert cfg.algo.name == "ppo"  # inherited
    assert cfg.algo.clip == 0.1  # file overrides base
    assert cfg.algo.epochs == 4  # CLI adds, parsed as int
    assert cfg.reader.name == "vllm"
    assert cfg.tags == ("a", "b")
    assert cfg.get_path("algo.missing", 7) == 7


def test_immutable_and_roundtrip():
    cfg = Config({"a": {"b": [1, {"c": 2}]}})
    with pytest.raises(TypeError):
        cfg.a = 1
    assert Config(cfg.to_dict()) == cfg
    assert "b:" in cfg.to_yaml()


def test_circular_base_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\n")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_repo_base_config_loads():
    from pathlib import Path

    cfg = load_config(Path(__file__).parents[1] / "configs" / "base.yaml")
    assert cfg.data.tier in {"strict", "relaxed"}
    assert cfg.env.horizon == 5
