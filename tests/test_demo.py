"""Inference demo core (mhtable/demo.py): traces for every selector kind, replay round trip,
run-id parsing / discovery, the reader swap, and the firewall (gold only after answering)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mhtable.config import load_config  # noqa: E402
from mhtable.demo import (Demo, SelectorSpec, baseline_specs, describe, discover_policies,  # noqa: E402
                          load_traces, parse_run_id, results_table, save_traces)
from mhtable.encoders import EmbCache, HashEncoder  # noqa: E402
from mhtable.featurise import Featuriser  # noqa: E402
from mhtable.policies import MLPPolicy  # noqa: E402
from mhtable.readers import MockReader, Reader  # noqa: E402
from mhtable.train import save_checkpoint  # noqa: E402

from test_env import GOLD, _env, _inst, data_dir  # noqa: E402,F401  (data_dir is a fixture)
from test_learning import gold  # noqa: E402,F401  (fixture)

BASE = Path(__file__).parents[1] / "configs" / "base.yaml"


@pytest.fixture
def demo(data_dir, tmp_path):  # noqa: F811
    cfg = load_config(BASE, ["policy.d=32", "policy.hidden=64"])
    env = _env(data_dir, tier="relaxed")
    cache = EmbCache.live(HashEncoder(32))
    torch.manual_seed(0)
    feat = Featuriser(env, cache)
    policy = MLPPolicy(feat.spec, d=32, hidden=64)
    run = tmp_path / "runs" / "ppo-r1-relaxed-lr1e-5-s0"
    run.mkdir(parents=True)
    save_checkpoint(run / "policy.pt", policy, feat.spec, cfg)
    d = Demo(cfg, reader=MockReader(), emb_cache=cache, live_encoder=False, envs={"relaxed": env})
    yield d, tmp_path / "runs"
    d.close()


def test_describe_is_plain_english():
    assert describe(GOLD[0]) == "Keep rows where `head.born_state` is 'Alabama'"
    assert describe(GOLD[1]).startswith("Join table `management` on `")
    assert describe(GOLD[-1]).startswith("Stop")


def test_parse_run_ids_of_every_script():
    assert parse_run_id("ppo-r1-strict-lr1e-5-s0") == ("PPO R1 lr1e-5", "grid", "strict")
    assert parse_run_id("grpoproc-r1-relaxed-lr1e-5-lam0.5-s0") == ("Process GRPO R1 λ0.5", "process", "relaxed")
    assert parse_run_id("grpo-r1-strict-bigx-s0") == ("GRPO R1 (bigx)", "bigstep", "strict")
    assert parse_run_id("ppo-r0-relaxed-scratch-s0") == ("PPO R0 from scratch", "scratch", "relaxed")
    assert parse_run_id("bc-bge-strict-s0") == ("BC", "bc", "strict")
    assert parse_run_id("something-new-strict-v2") == ("something-new-strict-v2", "other", "strict")


def test_discover_local_policies(demo):
    _, runs = demo
    (runs / "ppo-r1-strict-lr1e-5-s0-smoke").mkdir()
    (runs / "ppo-r1-strict-lr1e-5-s0-smoke" / "policy.pt").write_bytes(b"")
    specs = discover_policies(runs)
    assert [s.name for s in specs] == ["ppo-r1-relaxed-lr1e-5-s0"]  # smoke runs skipped
    assert specs[0].tier == "relaxed" and specs[0].kind == "policy"


def test_policy_trace_has_steps_probabilities_and_scored_answer(demo, gold):  # noqa: F811
    d, runs = demo
    spec = discover_policies(runs)[0]
    tr = d.run(spec, _inst(), gold, top_k=3)
    assert tr.steps and tr.steps[-1].kind == "STOP" or len(tr.steps) == d.env("relaxed").cfg.actions.horizon
    for st in tr.steps:
        assert 1 <= len(st.candidates) <= 3 and sum(c.chosen for c in st.candidates) == 1
        assert all(0.0 <= c.prob <= 1.0 for c in st.candidates)
        chosen = next(c for c in st.candidates if c.chosen)
        assert chosen.prob == max(c.prob for c in st.candidates)  # greedy
    assert tr.reader == "mock" and tr.em in (0.0, 1.0) and tr.gold_answer == "1903"
    assert tr.tokens <= tr.budget


def test_oracle_and_baselines(demo, gold):  # noqa: F811
    d, _ = demo
    specs = {s.name: s for s in baseline_specs()}
    orc = d.run(specs["oracle"], _inst(), gold, tier="relaxed")
    assert [s.kind for s in orc.steps] == [a.kind for a in GOLD] and orc.em == 1.0 and orc.uses_gold
    assert "WHERE" in orc.steps[0].sql_after and orc.steps[-1].sql_after == ""
    heu = d.run(specs["heuristic"], _inst(), gold, tier="relaxed")
    assert heu.steps and not heu.steps[0].candidates  # no probabilities for non-policy selectors
    b1 = d.run(specs["truncate"], _inst(), gold, tier="relaxed")
    assert b1.steps == [] and b1.rows_shown > 0 and b1.answer_in_subtable == 1.0
    with pytest.raises(ValueError):
        d.run(specs["oracle"], _inst(), None, tier="relaxed")  # the oracle needs gold
    with pytest.raises(ValueError):
        d.run(SelectorSpec("x", "x", "policy", tier="strict", path="p"), _inst(), gold, tier="relaxed")


def test_replay_round_trip(demo, gold, tmp_path):  # noqa: F811
    d, runs = demo
    traces = [d.run(discover_policies(runs)[0], _inst(), gold),
              d.run(baseline_specs()[1], _inst(), gold, tier="relaxed")]
    save_traces(tmp_path / "t.json", traces, {"reader": "mock"})
    back, meta = load_traces(tmp_path / "t.json")
    assert back == traces and meta == {"reader": "mock"}


class Echo(Reader):
    """Answers from the sub-table without gold: stands in for a real (swapped-in) LLM."""

    id = "echo"

    def read_batch(self, questions, subtables, meta=None):
        assert meta is None  # a real reader never receives gold
        return ["1903" if "1903" in s else "" for s in subtables]


def test_reader_swap_and_typed_question(demo, gold):  # noqa: F811
    d, runs = demo
    assert d.set_reader(reader=Echo()) == "echo"
    typed = Demo.typed_instance(_inst(), "Which departments are run by heads born in Alabama?")
    tr = d.run(discover_policies(runs)[0], typed, None)
    assert tr.reader == "echo" and tr.gold_answer is None and tr.em is None
    assert d.cfg.get_path("env.tokenizer") is None  # budget tokenizer pinned at creation
    d.set_reader({"name": "mock"})
    assert d.reader.id == "mock" and d.cfg.reader.name == "mock"


def test_results_table_local(tmp_path):
    import json

    for name, tiers in {"t-oracle-test": {"strict": 0.46}, "t-ppo-r1-strict-lr1e-5-s0-test": {"strict": 0.288},
                        "t-bc-relaxed-test": {"relaxed": 0.268}}.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text(json.dumps(
            {"tiers": {t: {"n": 10, "em": em, "tokens": 500} for t, em in tiers.items()}}))
    rows = {r["run"]: r for r in results_table(local_dir=tmp_path)}
    assert rows["oracle"]["group"] == "baseline" and rows["bc-relaxed"]["label"] == "BC"
    assert rows["ppo-r1-strict-lr1e-5-s0"]["label"] == "PPO R1 lr1e-5" and rows["ppo-r1-strict-lr1e-5-s0"]["em"] == 0.288
