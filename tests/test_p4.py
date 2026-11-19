"""P4: real readers (prompt, clean-up, vLLM call path with a fake engine), the reader's tokenizer in
materialise, the embedding-cache builder, the metric suite and the evaluation entry point.

Tests marked `network` load tiny Hugging Face models (transformers / sentence-transformers) and
are deselected by default: `pytest -m network`."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mhtable.encoders import EmbCache  # noqa: E402
from mhtable.env.tokens import CharEstimate  # noqa: E402
from mhtable.featurise import Featuriser, FeatConfig  # noqa: E402
from mhtable.metrics import answer_in_subtable, item_counts, paired_bootstrap, summarise, subtable_cells  # noqa: E402
from mhtable.readers.llm import chat_prompt, clean_answer, reader_id  # noqa: E402
from mhtable.rollout import collect  # noqa: E402
from mhtable.selectors.oracle import OracleSelector  # noqa: E402
from mhtable.readers import MockReader  # noqa: E402
from mhtable.sufficiency import entity_overlap, gold_hops  # noqa: E402
from mhtable.types import Action  # noqa: E402

from test_env import GOLD, _env, _inst, data_dir  # noqa: E402,F401  (data_dir is a fixture)
from test_learning import Scripted, _write_repo, gold  # noqa: E402,F401  (gold is a fixture)

ROOT = Path(__file__).parents[1]


class WordTok:
    """Stand-in tokenizer: one token per whitespace-separated word, no chat template."""

    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [t.split() for t in text]}
        return {"input_ids": text.split()}


class WordCounter:
    exact, id = True, "words"

    def count(self, text):
        return len(text.split())

    def count_many(self, texts):
        return [len(t.split()) for t in texts]


# ---------------------------------------------------------------- readers
def test_reader_id_and_answer_clean_up():
    a = reader_id("vllm", "Qwen/Qwen2.5-7B-Instruct", "float16", 512)
    assert a != reader_id("vllm", "Qwen/Qwen2.5-7B-Instruct", "bfloat16", 512) != reader_id("hf", "Qwen/Qwen2.5-7B-Instruct", "float16", 512)
    assert "float16" in a and "greedy" in a
    assert clean_answer("Answer: 1903") == "1903"
    assert clean_answer("```\n- State | 1789\n- Treasury | 1789\n```") == "State | 1789\nTreasury | 1789"
    assert clean_answer("\n\n  42  \n") == "42"


def test_chat_prompt_cuts_rows_to_fit():
    table = "\n".join(["| a |", "|---|"] + [f"| row{i} |" for i in range(50)])
    text, n, cut = chat_prompt(WordTok(), "how many?", table, 10_000)
    assert not cut and "row49" in text
    text, n, cut = chat_prompt(WordTok(), "how many?", table, n - 20)
    assert cut and n <= len(text.split()) and "| a |" in text and "row49" not in text and "row0" in text


def test_vllm_reader_call_path(monkeypatch):
    seen = {}

    class LLM:
        def __init__(self, **kw):
            seen["init"] = kw

        def get_tokenizer(self):
            return WordTok()

        def generate(self, prompts, params, use_tqdm=False):
            seen["prompts"], seen["params"] = prompts, params
            out = lambda t, f: types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t, finish_reason=f)])  # noqa: E731
            return [out("Answer: 1903", "stop"), out("a\nb", "length")]

    class SamplingParams:
        def __init__(self, temperature, max_tokens):
            self.temperature, self.max_tokens = temperature, max_tokens

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams))
    from mhtable.registry import build

    r = build("reader", {"name": "vllm", "model": "m", "dtype": "float16", "batch_size": 4})  # hf-only key ignored
    assert r.read_batch(["q1?", "q2?"], ["| x |\n|---|\n| 1 |", "| y |"]) == ["1903", "a\nb"]
    assert seen["params"].temperature == 0.0 and seen["init"]["dtype"] == "float16"
    assert "q1?" in seen["prompts"][0] and "q2?" in seen["prompts"][1]
    st = r.stats.to_dict()
    assert st["prompts"] == 2 and st["answers_cut"] == 1 and st["prompts_cut"] == 0
    with pytest.raises(TypeError):
        build("reader", {"name": "vllm", "model": "m", "temprature": 0})
    r2 = build("reader", {"name": "vllm", "model": "m", "dtype": "float16", "prompt": "p2"})
    r2.read_batch(["q1?", "q2?"], ["| x |", "| y |"])
    assert r2.id.endswith(":p2") and r.id.endswith(":p1") and "Do not stop early" in seen["prompts"][0]
    assert r2.stats.to_dict()["mean_prompt_tokens"] > 0
    from mhtable.readers import CachedReader

    assert CachedReader(r).key("q", "| x |") != CachedReader(r2).key("q", "| x |")
    with pytest.raises(ValueError):
        build("reader", {"name": "vllm", "model": "m", "prompt": "p9"})


# ---------------------------------------------------------------- tokenizer in materialise
def test_materialise_with_an_exact_token_counter(data_dir):  # noqa: F811
    env = _env(data_dir)
    assert isinstance(env.counter, CharEstimate)  # default: unchanged character estimate
    env.counter = WordCounter()
    s = env.reset(_inst())
    s, _ = env.step(s, Action("SEED", "department"))
    full = env.materialise(s)
    assert full.rows_shown == 3 and not full.overflow and full.tokens == len(full.text.split())
    cut = env.materialise(s, budget=full.tokens - 1)
    assert cut.overflow and cut.rows_shown == 2 and cut.tokens <= full.tokens - 1
    env.close()


# ---------------------------------------------------------------- metrics
def test_answer_coverage_items_and_bootstrap():
    sub = "| department.creation | head.born_state |\n|---|---|\n| 1903 | Alabama |\n| 1789.0 |  |"
    assert subtable_cells(sub) == {"1903", "alabama", "1789"}
    assert answer_in_subtable(sub, [["1903"]]) == 1.0 and answer_in_subtable(sub, [["1903"], ["2001"]]) == 0.5
    c = item_counts([GOLD[1].edge], [GOLD[0].anchor], [GOLD[1].edge, GOLD[2].edge], [GOLD[0].anchor])
    assert (c["edge_hit"], c["edge_pred"], c["edge_gold"], c["anchor_hit"]) == (1, 1, 2, 1)
    rows = [{"source": "spider", "tables": "3_tables", "category": "List", "monotone": True, "em": 1.0,
             "answer_in_subtable": 1.0, "edge_hit": 1, "edge_pred": 1, "edge_gold": 2,
             "hops": [{"hop": 0, "recall": 1.0, "precision": 0.5}]},
            {"source": "mmqa", "tables": "2_tables", "category": "Count", "monotone": False, "em": 0.0,
             "answer_in_subtable": 0.0, "edge_hit": 0, "edge_pred": 1, "edge_gold": 1, "hops": []}]
    s = summarise(rows)
    assert s["n"] == 2 and s["em"] == 0.5 and s["em_when_answer_shown"] == 1.0 and s["edge_recall"] == round(1 / 3, 4)
    assert s["by_source"]["mmqa"]["em"] == 0.0 and s["per_hop"]["hop0"] == {"recall": 1.0, "precision": 0.5}
    b = paired_bootstrap([1, 1, 1, 0] * 25, [0, 1, 0, 0] * 25, n=2000)
    assert b["diff"] == 0.5 and b["ci95"][0] > 0 and b["p"] < 0.01
    assert paired_bootstrap([1, 0] * 10, [1, 0] * 10, n=500)["p"] == 1.0


def test_per_hop_entity_overlap(data_dir, gold):  # noqa: F811
    env = _env(data_dir)
    assert gold_hops(gold) == {"head": 0, "management": 1, "department": 2}
    s = env.reset(_inst())
    for a in GOLD[:-1]:
        s, _ = env.step(s, next(c for c in env.candidate_actions(s) if c.key() == a.key()))
    full = {h["table"]: h for h in entity_overlap(env, s, gold)}
    assert all(h["recall"] == 1.0 and h["precision"] == 1.0 for h in full.values())
    s = env.reset(_inst())  # the anchor and one join only: department never reached
    for a in GOLD[:2]:
        s, _ = env.step(s, next(c for c in env.candidate_actions(s) if c.key() == a.key()))
    part = {h["table"]: h for h in entity_overlap(env, s, gold)}
    assert part["department"]["recall"] == 0.0 and part["department"]["precision"] is None
    assert part["head"]["recall"] == 1.0
    env.close()


def test_rollout_detail_metrics(data_dir, gold):  # noqa: F811
    env = _env(data_dir)
    golds = {gold.instance_id: gold}
    (tr,) = collect(env, OracleSelector(golds), MockReader(), [_inst()], mode="argmax", oracle_mode="metric_only",
                    golds=golds, detail=True)
    i = tr.info
    assert i["r_ans"] == 1.0 and i["set_f1"] == 1.0 and i["answer_in_subtable"] == 1.0
    assert i["edge_hit"] == i["edge_gold"] == 2 and i["anchor_hit"] == 1 and len(i["hops"]) == 3
    (plain,) = collect(env, OracleSelector(golds), MockReader(), [_inst()], mode="argmax", oracle_mode="metric_only",
                       golds=golds)
    assert "set_f1" not in plain.info  # training rollouts skip the extra queries
    env.close()


# ---------------------------------------------------------------- embedding cache
def test_build_cache_covers_every_featuriser_lookup(tmp_path, data_dir, gold):  # noqa: F811
    from mhtable.encoders.build_cache import main

    _write_repo(data_dir, gold)
    out = tmp_path / "emb"
    assert main(["--family", "spider_mmqa", "--encoder", "hash", "--data-dir", str(data_dir), "--out", str(out),
                 "--chunk", "7"]) == 0
    final = out / "hash64" / "spider_mmqa.npz"
    manifest = json.loads((out / "hash64" / "spider_mmqa.json").read_text())
    cache = EmbCache.load(final)
    assert manifest["texts"] == len(cache) and manifest["questions"] == 1 and cache.dim == 64
    assert cache._matrix.dtype == np.float16 and cache.get(_inst().question).dtype == np.float32
    assert main(["--family", "spider_mmqa", "--encoder", "hash", "--data-dir", str(data_dir), "--out", str(out),
                 "--chunk", "7"]) == 0  # rerun: every part already exists and is reused
    for ablate in (False, True):  # a frozen cache: any miss raises
        env = _env(data_dir)
        feat = Featuriser(env, cache, FeatConfig(header_ablation=ablate))
        s = env.reset(_inst())
        for a in GOLD:
            cands = env.candidate_actions(s)
            feat(s, cands)
            s, done = env.step(s, next(c for c in cands if c.key() == a.key()))
        env.close()


# ---------------------------------------------------------------- evaluation entry point
def test_evaluate_entry_point(tmp_path, data_dir, gold):  # noqa: F811
    from mhtable import evaluate

    _write_repo(data_dir, gold)
    base = ["--config", str(ROOT / "configs" / "experiments" / "eval.yaml"), "--data-dir", str(data_dir),
            "--out", str(tmp_path / "runs"), "data.sources=[spider]", "data.revision=null", "env.tokenizer=null",
            "reader={name: mock}"]
    assert evaluate.main(base + ["--selector", "oracle", "--run-id", "o"]) == 0
    res = json.loads((tmp_path / "runs" / "o" / "result.json").read_text())
    assert set(res["tiers"]) == {"strict", "relaxed"} and res["reportable"] is False
    for t in res["tiers"].values():
        assert t["n"] == 1 and t["em"] == 1.0 and t["path_match"] == 1.0 and t["answer_in_subtable"] == 1.0
        assert t["by_tables"]["3_tables"]["em"] == 1.0
    pred = json.loads((tmp_path / "runs" / "o" / "predictions_relaxed.jsonl").read_text().splitlines()[0])
    assert pred["answer"] == "1903" and pred["gold_answer"] == [["1903"]]
    with pytest.raises(SystemExit):
        evaluate.main(base + ["--selector", "oracle", "--run-id", "r", "--reported"])


# ---------------------------------------------------------------- tiny real models (network)
@pytest.mark.network
def test_hf_reader_and_tokenizer_with_a_tiny_model(data_dir):  # noqa: F811
    from mhtable.env.tokens import HFTokenCounter
    from mhtable.registry import build

    model = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    r = build("reader", {"name": "hf", "model": model, "dtype": "float32", "max_new_tokens": 8, "batch_size": 2,
                         "max_model_len": 6144})  # a vllm-only key is ignored
    out = r.read_batch(["q?", "a much longer question?", "q3?"], ["| a |\n|---|\n| 1 |"] * 3)
    assert len(out) == 3 and all(isinstance(x, str) for x in out) and r.stats.prompts == 3
    env = _env(data_dir)
    env.counter = HFTokenCounter(model)
    s = env.reset(_inst())
    s, _ = env.step(s, Action("SEED", "department"))
    m = env.materialise(s, budget=30)
    assert m.tokens == env.counter.count(m.text) <= 30
    env.close()


@pytest.mark.network
def test_st_encoder_with_a_tiny_model():
    from mhtable.registry import build

    enc = build("encoder", {"name": "st", "model": "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                            "dim": 64})  # base.yaml's dim is ignored: the model fixes it
    v = enc.encode(["head born state", "department creation"])
    assert v.shape == (2, enc.dim) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert enc.id == "stsb-bert-tiny-safetensors"
