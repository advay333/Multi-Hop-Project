"""P5 gate: shaping telescopes, GAE at gamma = 1, masking, firewall — plus the shared rollout,
the mock reader, the reader cache and a small learning sanity check."""

import ast
import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mhtable.algos.base import collate, evaluate_actions, gae, reward_to_go  # noqa: E402
from mhtable.data.execute import connect_ro  # noqa: E402
from mhtable.data.schema import SchemaGraph  # noqa: E402
from mhtable.encoders import EmbCache, HashEncoder  # noqa: E402
from mhtable.featurise import FeatConfig, Featuriser, required_texts  # noqa: E402
from mhtable.firewall import FirewallError  # noqa: E402
from mhtable.metrics import em_norm, set_f1  # noqa: E402
from mhtable.oracle import parse, run_oracle  # noqa: E402
from mhtable.policies import MLPPolicy  # noqa: E402
from mhtable.policies.base import MASKED  # noqa: E402
from mhtable.readers import CachedReader, MockReader, Reader  # noqa: E402
from mhtable.readers.prompt import build_prompt  # noqa: E402
from mhtable.rewards import RewardConfig  # noqa: E402
from mhtable.rollout import Decision, PolicyActor, collect  # noqa: E402
from mhtable.types import GoldRecord  # noqa: E402

from conftest import QUESTIONS, SCHEMA_ENTRY  # noqa: E402
from test_env import GOLD, REF, _env, _inst, data_dir  # noqa: E402,F401  (data_dir is a fixture)

SRC = Path(__file__).parents[1] / "src" / "mhtable"


@pytest.fixture
def gold(data_dir):  # noqa: F811
    """The real oracle's record for the Alabama question on the fixture database."""
    schema = SchemaGraph.from_tables_json(SCHEMA_ENTRY)
    con = connect_ro(data_dir / "db" / f"{REF}.sqlite")
    parsed, _ = parse(QUESTIONS[0]["query"], schema)
    res = run_oracle(parsed, schema, con)
    con.close()
    assert not res.fail_reasons
    return GoldRecord(_inst().instance_id, QUESTIONS[0]["query"], [["1903"]], gold_tables=parsed.tables,
                      gold_edges=parsed.edges, gold_anchors=parsed.anchors, gold_entity_sets=res.entity_sets,
                      gold_trajectory=res.trajectory, monotone=parsed.monotone)


@pytest.fixture
def setup(data_dir, gold):  # noqa: F811
    env = _env(data_dir)
    feat = Featuriser(env, EmbCache.live(HashEncoder(32)))
    torch.manual_seed(0)
    policy = MLPPolicy(feat.spec, d=32, hidden=64)
    yield env, feat, policy, {gold.instance_id: gold}
    env.close()


class Scripted:
    """Actor that follows a fixed list of actions (matched by key), then STOPs."""

    def __init__(self, plan):
        self.plan = plan

    def act(self, states, cands, mode, generator=None):
        out = []
        for s, c in zip(states, cands):
            want = self.plan[s.t].key() if s.t < len(self.plan) else ("STOP", None, None, None)
            out.append(Decision(next(i for i, a in enumerate(c) if a.key() == want)))
        return out


class RandomActor:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, states, cands, mode, generator=None):
        return [Decision(self.rng.randrange(len(c))) for c in cands]


# ---------------------------------------------------------------- rewards
def test_worked_example_rewards(setup):
    env, _, _, golds = setup
    cfg = RewardConfig("R1", beta=0.5, lambda_cost=0.1)
    (tr,) = collect(env, Scripted(GOLD), MockReader(), [_inst()], oracle_mode="reward", golds=golds, reward_cfg=cfg)
    assert [round(s.phi, 4) for s in tr.steps] == [0.0, 0.3333, 0.6667, 1.0]  # report Figure 1
    cost = 0.1 * tr.subtable_tokens / env.cfg.budget_tokens
    expect = [0.5 / 3, 0.5 / 3, 0.5 / 3, -0.5 + 1.0 - cost]
    assert [s.reward for s in tr.steps] == pytest.approx(expect)
    assert tr.info["path_match"] and tr.info["sufficient"] and tr.info["r_ans"] == 1.0
    assert tr.answer == "1903" and tr.reader_called


def test_shaping_telescopes(setup):
    env, _, _, golds = setup
    shaped = RewardConfig("R1", beta=0.5, lambda_cost=0.1)
    sparse = RewardConfig("R0", beta=0.5, lambda_cost=0.1)
    for seed in range(3):
        a = collect(env, RandomActor(seed), MockReader(), [_inst()] * 20, oracle_mode="reward", golds=golds,
                    reward_cfg=shaped)
        b = collect(env, RandomActor(seed), MockReader(), [_inst()] * 20, oracle_mode="reward", golds=golds,
                    reward_cfg=sparse)
        for x, y in zip(a, b):
            outcome = x.info["r_ans"] - 0.1 * x.subtable_tokens / env.cfg.budget_tokens
            assert sum(s.reward for s in x.steps) == pytest.approx(outcome, abs=1e-9)  # shaping sums to 0
            assert sum(s.reward for s in y.steps) == pytest.approx(outcome, abs=1e-9)  # same return under R0
            assert all(s.reward == 0 for s in y.steps[:-1])
        assert any(s.reward != 0 for x in a for s in x.steps[:-1])  # the potential does discriminate steps


def test_curriculum_starts_after_gold_steps(setup):
    env, _, _, golds = setup
    cfg = RewardConfig("R1", beta=0.5, lambda_cost=0.0)
    (tr,) = collect(env, Scripted(GOLD), MockReader(), [_inst()], oracle_mode="reward", golds=golds,
                    reward_cfg=cfg, start_after_gold_steps=2)
    assert len(tr.steps) == 2 and tr.info["start_t"] == 2 and tr.steps[0].phi == pytest.approx(2 / 3)
    assert sum(s.reward for s in tr.steps) == pytest.approx(1.0 - 0.5 * 2 / 3)  # R_ans - beta * Phi(start)
    with pytest.raises(ValueError, match="curriculum"):
        collect(env, Scripted(GOLD), MockReader(), [_inst()], oracle_mode="metric_only", golds=golds,
                start_after_gold_steps=1)


# ---------------------------------------------------------------- advantages
def test_gae_at_gamma_one():
    r, v = [0.1, -0.2, 0.0, 1.0], [0.5, 0.4, 0.7, 0.9]
    rtg = reward_to_go(r)
    assert rtg == pytest.approx([0.9, 0.8, 1.0, 1.0])
    adv, ret = gae(r, v, gamma=1.0, lam=1.0)  # lambda = 1: Monte-Carlo return minus the baseline
    assert adv == pytest.approx([g - x for g, x in zip(rtg, v)]) and ret == pytest.approx(rtg)
    adv0, _ = gae(r, v, gamma=1.0, lam=0.0)  # lambda = 0: one-step TD error, V(terminal) = 0
    assert adv0 == pytest.approx([0.1 + 0.4 - 0.5, -0.2 + 0.7 - 0.4, 0.0 + 0.9 - 0.7, 1.0 - 0.9])
    adv95, ret95 = gae(r, v, gamma=1.0, lam=0.95)
    deltas = adv0
    assert adv95[1] == pytest.approx(deltas[1] + 0.95 * deltas[2] + 0.95 ** 2 * deltas[3])
    assert ret95 == pytest.approx([a + x for a, x in zip(adv95, v)])
    assert gae(r, [0.9, 0.8, 1.0, 1.0], lam=0.95)[0] == pytest.approx([0, 0, 0, 0], abs=1e-9)  # perfect critic


# ---------------------------------------------------------------- policy
def test_masking_and_permutation_invariance(setup):
    env, feat, policy, _ = setup
    s = env.reset(_inst())
    cands = env.candidate_actions(s)
    sf, af = feat(s, cands)
    n = len(cands)
    state = torch.from_numpy(sf)[None]
    actions = torch.zeros(1, n + 5, af.shape[1])
    actions[0, :n] = torch.from_numpy(af)
    actions[0, n:] = 7.0  # garbage in the padding must not matter
    mask = torch.zeros(1, n + 5, dtype=torch.bool)
    mask[0, :n] = True
    logits, value = policy(state, actions, mask)
    assert (logits[0, n:] == MASKED).all() and value.shape == (1,)
    probs = torch.softmax(logits, -1)[0]
    assert probs[n:].sum().item() == 0.0 and probs[:n].sum().item() == pytest.approx(1.0, abs=1e-5)
    plain, _ = policy(state, torch.from_numpy(af)[None], torch.ones(1, n, dtype=torch.bool))
    assert torch.allclose(plain[0], logits[0, :n], atol=1e-6)  # padding does not change real logits
    perm = torch.randperm(n)
    shuffled, _ = policy(state, torch.from_numpy(af)[None][:, perm], torch.ones(1, n, dtype=torch.bool))
    assert torch.allclose(shuffled[0], plain[0, perm], atol=1e-6)  # candidate order is irrelevant
    assert probs[:n].max().item() < 2.0 / n  # small final-layer init: near uniform at the start
    g = torch.Generator().manual_seed(0)
    assert all(d.action_idx < len(c) for _ in range(50)
               for d, c in zip(PolicyActor(policy, feat).act([s], [cands], "sample", g), [cands]))


def test_policy_rollout_collate_and_groups(setup):
    env, feat, policy, golds = setup
    g = torch.Generator().manual_seed(0)
    trajs = collect(env, PolicyActor(policy, feat), MockReader(), [_inst()] * 2, mode="sample",
                    oracle_mode="reward", golds=golds, group_size=4, generator=g)
    assert len(trajs) == 8 and [t.info["group_member"] for t in trajs] == [0, 1, 2, 3] * 2
    batch = collate(trajs)
    n = sum(len(t.steps) for t in trajs)
    assert len(batch) == n and batch.state.shape == (n, feat.state_dim)
    assert batch.actions.shape[0] == n and batch.actions.shape[2] == feat.action_dim
    assert batch.group.max().item() == 0 and batch.traj.max().item() == 7  # one question, eight rollouts
    assert (batch.action_idx < batch.mask.sum(1)).all()
    logp, entropy, value, _ = evaluate_actions(policy, batch)
    assert torch.allclose(logp, batch.logp, atol=1e-5) and torch.allclose(value, batch.value, atol=1e-5)
    assert (entropy >= -1e-6).all() and entropy.max() > 1.0  # 0 only where STOP is the single candidate
    # argmax is deterministic; "off" uses no gold and so refuses the mock reader
    a = collect(env, PolicyActor(policy, feat), MockReader(), [_inst()], mode="argmax", oracle_mode="metric_only",
                golds=golds)
    b = collect(env, PolicyActor(policy, feat), MockReader(), [_inst()], mode="argmax", oracle_mode="metric_only",
                golds=golds)
    assert a[0].info["actions"] == b[0].info["actions"] and all(s.reward == 0 for s in a[0].steps)
    with pytest.raises(ValueError, match="no gold"):
        collect(env, PolicyActor(policy, feat), MockReader(), [_inst()], oracle_mode="off")


def test_bc_style_fit_follows_the_gold_path(setup):
    """Features carry enough signal: a few cross-entropy steps on the gold actions make the
    greedy policy reproduce the gold trajectory (the P7 behaviour-cloning update in miniature)."""
    env, feat, policy, golds = setup
    trajs = collect(env, Scripted(GOLD), MockReader(), [_inst()], oracle_mode="reward", golds=golds)
    s = env.reset(_inst())
    for step in trajs[0].steps:  # re-featurise the gold path (the scripted actor stores no features)
        cands = env.candidate_actions(s)
        step.state_feat, step.action_feats = feat(s, cands)
        s, _ = env.step(s, cands[step.action_idx])
    batch = collate(trajs)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    for _ in range(40):
        logp, _, _, _ = evaluate_actions(policy, batch)
        loss = -logp.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    (tr,) = collect(env, PolicyActor(policy, feat), MockReader(), [_inst()], mode="argmax",
                    oracle_mode="metric_only", golds=golds)
    assert tr.info["path_match"] and tr.info["r_ans"] == 1.0


# ---------------------------------------------------------------- mock reader, sufficiency, cache
def test_mock_reader_follows_sufficiency(setup):
    env, _, _, golds = setup
    seed_dept = [a for a in env.candidate_actions(env.reset(_inst())) if a.kind == "SEED" and a.table == "department"]
    (bad,) = collect(env, Scripted(seed_dept), MockReader(), [_inst()], oracle_mode="metric_only", golds=golds)
    assert not bad.info["sufficient"] and bad.info["r_ans"] == 0.0 and bad.answer == ""
    # No filter, all three tables joined: a different path whose rows still contain the gold rows.
    s = env.reset(_inst())
    plan = [next(a for a in env.candidate_actions(s) if a.kind == "SEED" and a.table == "head"), GOLD[1], GOLD[2]]
    (sup,) = collect(env, Scripted(plan), MockReader(), [_inst()], oracle_mode="metric_only", golds=golds)
    assert sup.info["sufficient"] and not sup.info["path_match"] and sup.info["r_ans"] == 1.0
    golds[_inst().instance_id].monotone = False  # a counting question needs exactly the gold rows
    (cnt,) = collect(env, Scripted(plan), MockReader(), [_inst()], oracle_mode="metric_only", golds=golds)
    assert not cnt.info["sufficient"]


def test_reader_cache_counts_only_misses(tmp_path):
    class Echo(Reader):
        id = "echo"

        def __init__(self):
            self.batches = []

        def read_batch(self, questions, subtables, meta=None):
            self.batches.append(len(questions))
            return [q.upper() for q in questions]

    shard = tmp_path / "reader" / "echo" / "run.jsonl"
    r = CachedReader(Echo(), shard)
    ans, called = r.read_batch(["a", "b", "a"], ["T", "T", "T"])
    assert ans == ["A", "B", "A"] and called == [True, True, False] and (r.calls, r.hits) == (2, 1)
    ans, called = r.read_batch(["b", "c"], ["T", "T"])
    assert called == [False, True] and (r.calls, r.hits) == (3, 2) and r.reader.batches == [2, 1]
    again = CachedReader(Echo(), shard)  # a new run starts from the shard on disk
    assert again.read_batch(["a", "c"], ["T", "T"])[1] == [False, False] and again.calls == 0
    assert r.key("a", "T") != r.key("a", "T2") and len(r.key("a", "T")) == 64


def test_answer_matching():
    assert em_norm("1903", [["1903"]]) == 1.0 and em_norm(" 1903.0 \n", [["1903"]]) == 1.0
    assert em_norm("Alabama | 7\ncalifornia | 2", [["california", "2"], ["alabama", "7"]]) == 1.0  # row order free
    assert em_norm("alabama\ncalifornia", [["california"], ["alabama"]]) == 1.0
    assert em_norm("alabama", [["california"], ["alabama"]]) == 0.0 and em_norm("", [["1"]]) == 0.0
    assert set_f1("alabama", [["california"], ["alabama"]]) == pytest.approx(2 / 3)
    from mhtable.metrics import format_answer
    for gold in ([[""]], [["laurel", ""]], [["a|b", "x\ny"]], [["2012-01-05", "7.0"], ["<null>", "1,234"]]):
        assert em_norm(format_answer(gold), gold) == 1.0  # what the mock reader writes always scores 1


# ---------------------------------------------------------------- features and firewall
def test_features_shapes_cache_and_header_ablation(setup, data_dir):  # noqa: F811
    env, feat, _, _ = setup
    s = env.reset(_inst())
    cands = env.candidate_actions(s)
    sf, af = feat(s, cands)
    assert sf.shape == (feat.state_dim,) and af.shape == (len(cands), feat.action_dim)
    assert np.isfinite(sf).all() and np.isfinite(af).all()
    assert np.abs(af[:, 3 * feat.d_emb:]).max() <= 1.0 + 1e-6  # scalars are scaled
    # A frozen cache built from required_texts covers every lookup the featuriser makes.
    texts = required_texts(_inst(), env.data.schema(REF), env.data.index(REF)[0])
    frozen = EmbCache(32)
    frozen.add(texts, HashEncoder(32).encode(texts))
    strict = Featuriser(env, frozen)
    while True:
        s2f, a2f = strict(s, cands)
        assert s2f.shape == sf.shape
        s, done = env.step(s, cands[-1] if cands[-1].kind == "STOP" else cands[0])
        if done:
            break
        cands = env.candidate_actions(s)
    with pytest.raises(KeyError, match="cache miss"):
        Featuriser(env, EmbCache(32))(env.reset(_inst()), env.candidate_actions(env.reset(_inst())))
    s = env.reset(_inst())
    cands = env.candidate_actions(s)
    ablated = Featuriser(env, EmbCache.live(HashEncoder(32)), FeatConfig(header_ablation=True))
    _, af2 = ablated(s, cands)
    assert af2.shape == af.shape and not np.allclose(af2, feat(s, cands)[1])
    assert "born_state" not in " ".join(ablated.cache.vectors) and "col_" in " ".join(ablated.cache.vectors)


def test_firewall_rejects_gold(setup, gold):
    env, feat, _, _ = setup
    s = env.reset(_inst())
    s.instance = gold  # a gold record where the public instance belongs
    with pytest.raises(FirewallError):
        feat.state_features(s)
    with pytest.raises(FirewallError):
        build_prompt("q", gold)
    assert "1903" not in build_prompt("q?", "| a |") and "q?" in build_prompt("q?", "| a |")


def test_gold_free_modules_never_import_gold():
    """Featuriser, policies, encoders, the firewall helper and the reader prompt/cache see no gold."""
    files = [SRC / "featurise.py", SRC / "firewall.py", SRC / "readers" / "prompt.py", SRC / "readers" / "base.py"]
    files += list((SRC / "policies").glob("*.py")) + list((SRC / "encoders").glob("*.py"))
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported |= {node.module or ""} | {f"{node.module}.{a.name}" for a in node.names}
            elif isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
        bad = [m for m in imported if any(x in m for x in ("oracle", "GoldRecord", "rewards", "sufficiency", "mock"))]
        assert not bad, f"{f.name} imports {bad}"


# ---------------------------------------------------------------- P7: behaviour cloning
def _write_repo(data_dir, gold, question_ids=("train", "dev")):  # noqa: F811
    """Add instances / gold / splits to the miniature data repo (one question in train and dev)."""
    import json

    inst = _inst()
    inst.tier = {"strict": True, "relaxed": True}
    inst.n_gold_tables = 3
    (data_dir / "instances").mkdir(exist_ok=True)
    (data_dir / "gold").mkdir(exist_ok=True)
    (data_dir / "instances" / "spider.jsonl").write_text(json.dumps(inst.to_dict()) + "\n", encoding="utf-8")
    (data_dir / "gold" / "spider.jsonl").write_text(json.dumps(gold.to_dict()) + "\n", encoding="utf-8")
    (data_dir / "splits" / "spider_mmqa").mkdir(parents=True, exist_ok=True)
    for split in question_ids:
        (data_dir / "splits" / "spider_mmqa" / f"{split}.txt").write_text(inst.instance_id + "\n")


def test_load_split_and_demonstrations(setup, data_dir, gold):  # noqa: F811
    from mhtable.algos.bc import BC, demonstrations
    from mhtable.dataset import load_split

    env, feat, policy, _ = setup
    _write_repo(data_dir, gold)
    items = load_split(env.data, "train", "relaxed", ("spider",))
    assert len(items) == 1 and items[0][1].gold_trajectory[-1].kind == "STOP"
    items[0][0].tier["strict"] = False
    demos, stats = demonstrations(env, feat, items)
    assert stats["reachable"] == 1 and stats["steps"] == 4 and len(demos[0].steps) == 4
    # A gold action the environment does not offer -> no demonstration, counted.
    broken = GoldRecord.from_dict(gold.to_dict())
    broken.gold_trajectory[0] = GOLD[2]
    assert demonstrations(env, feat, [(items[0][0], broken)])[1]["unreachable"] == 1
    bc = BC(policy, lr=1e-2, batch_size=2)
    batch = bc.compute_advantages(demos)
    critic_before = [p.clone() for p in policy.critic_parameters()]
    first = bc.update(batch)
    for _ in range(40):
        last = bc.update(batch)
    assert last["loss"] < first["loss"] and bc.evaluate(batch)["traj_acc"] == 1.0
    assert all(torch.equal(a, b) for a, b in zip(critic_before, policy.critic_parameters()))  # actor only


def test_train_entry_point_bc(tmp_path, data_dir, gold):  # noqa: F811
    import json

    from mhtable import train

    _write_repo(data_dir, gold)
    cfg = Path(__file__).parents[1] / "configs" / "experiments" / "bc.yaml"
    assert train.main(["--config", str(cfg), "--data-dir", str(data_dir), "--out", str(tmp_path / "runs"),
                       "--run-id", "t", "data.sources=[spider]", "policy.d=32", "policy.hidden=64",
                       "encoder.dim=32", "algo.lr=0.01", "train.epochs=40", "train.eval_every=20"]) == 0
    run = tmp_path / "runs" / "t"
    result = json.loads((run / "result.json").read_text())
    assert result["dev"]["path_match"] == 1.0 and result["dev"]["em"] == 1.0 and result["dev_untrained"]["n"] == 1
    assert result["demonstrations"]["train"]["reachable"] == 1 and result["train_subset"]["path_match"] == 1.0
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 2 and (run / "config.yaml").exists()
    policy, ck = train.load_checkpoint(run / "policy.pt")
    assert ck["spec"]["d_emb"] == 32 and policy.n_params() > 0


# ---------------------------------------------------------------- P8: PPO
def test_ppo_advantages_and_update(setup):
    import copy

    from mhtable.algos.ppo import PPO

    env, feat, policy, golds = setup
    g = torch.Generator().manual_seed(0)
    trajs = collect(env, PolicyActor(policy, feat), MockReader(), [_inst()] * 16, mode="sample",
                    oracle_mode="reward", golds=golds, generator=g)
    ppo = PPO(policy, bc_policy=copy.deepcopy(policy), batch_size=32, epochs=2)
    batch = ppo.compute_advantages(trajs)
    adv, ret = gae([s.reward for s in trajs[0].steps], [s.value for s in trajs[0].steps], 1.0, 0.95)
    n0 = len(trajs[0].steps)
    assert batch.advantage[:n0].tolist() == pytest.approx(adv) and batch.returns[:n0].tolist() == pytest.approx(ret)
    before = [p.clone() for p in policy.parameters()]
    stats = ppo.update(batch)
    assert all(np.isfinite(v) for v in stats.values()) and stats["kl_bc"] >= -1e-6 and stats["entropy"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
    assert all(not p.requires_grad for p in ppo.bc_policy.parameters())  # the BC anchor is frozen
    ppo.set_progress(1.0)
    assert ppo.entropy_coef == pytest.approx(0.001)
    assert PPO(policy).kl_bc_coef == 0.0  # no BC policy -> no KL term


def test_ppo_target_kl_stops_early_and_temperature_can_be_frozen(setup):
    from mhtable.algos.ppo import PPO

    env, feat, policy, golds = setup
    g = torch.Generator().manual_seed(0)
    trajs = collect(env, PolicyActor(policy, feat), MockReader(), [_inst()] * 16, mode="sample",
                    oracle_mode="reward", golds=golds, generator=g)
    t0 = float(policy.log_temperature)
    ppo = PPO(policy, lr_actor=0.05, batch_size=4, epochs=20, target_kl=1e-6, freeze_temperature=True)
    stats = ppo.update(ppo.compute_advantages(trajs))
    assert stats["epochs_run"] < 20  # a big learning rate trips the KL stop
    assert float(policy.log_temperature) == t0 and not policy.log_temperature.requires_grad
    assert all(np.isfinite(v) for v in stats.values())


# ---------------------------------------------------------------- P9: GRPO
def test_grpo_group_advantages_and_update(setup):
    import copy

    from mhtable.algos.grpo import GRPO

    env, feat, policy, golds = setup
    g = torch.Generator().manual_seed(0)
    trajs = collect(env, PolicyActor(policy, feat), MockReader(), [_inst()] * 2, mode="sample",
                    oracle_mode="reward", golds=golds, group_size=4, generator=g)
    algo = GRPO(policy, bc_policy=copy.deepcopy(policy), group_size=4, batch_size=32, epochs=2)
    assert algo.group_size == 4
    critic = {id(p) for p in policy.critic_parameters()}
    assert not any(id(p) in critic for grp in algo.opt.param_groups for p in grp["params"])  # no critic
    batch = algo.compute_advantages(trajs)
    for k in (0, 4):  # each block of 4 is one group: advantages are its normalised returns
        rets = np.array([sum(s.reward for s in tr.steps) for tr in trajs[k:k + 4]])
        want = np.zeros(4) if rets.std() < 1e-6 else (rets - rets.mean()) / (rets.std() + 1e-6)
        start = sum(len(tr.steps) for tr in trajs[:k])
        got = [float(batch.advantage[start + sum(len(t.steps) for t in trajs[k:k + i])]) for i in range(4)]
        assert got == pytest.approx(want.tolist(), abs=1e-5)
    critic_before = [p.clone() for p in policy.critic_parameters()]
    stats = algo.update(batch)
    assert "value_loss" not in stats and 0.0 <= stats["zero_adv_frac"] <= 1.0
    assert all(np.isfinite(v) for v in stats.values())
    assert all(torch.equal(a, b) for a, b in zip(critic_before, policy.critic_parameters()))


def test_grpo_identical_returns_give_zero_advantage(setup):
    from mhtable.algos.grpo import GRPO

    env, feat, policy, golds = setup
    trajs = collect(env, PolicyActor(policy, feat), MockReader(), [_inst()], mode="argmax",
                    oracle_mode="reward", golds=golds, group_size=4)  # greedy: 4 identical episodes
    algo = GRPO(policy, group_size=4)
    batch = algo.compute_advantages(trajs)
    assert torch.count_nonzero(batch.advantage) == 0 and algo._group_stats["zero_adv_frac"] == 1.0
    with pytest.raises(ValueError):
        algo.compute_advantages(trajs[:3])  # not a whole group
    with pytest.raises(ValueError):
        GRPO(policy, group_size=1)


def test_train_entry_point_grpo_overfits_one_question(tmp_path, data_dir, gold):  # noqa: F811
    """G-8 in miniature for GRPO: random start, mock reader, groups of 8."""
    import json

    from mhtable import train

    _write_repo(data_dir, gold)
    cfg = Path(__file__).parents[1] / "configs" / "experiments" / "grpo.yaml"
    args = ["--config", str(cfg), "--data-dir", str(data_dir), "--out", str(tmp_path / "runs"), "--run-id", "g",
            "data.sources=[spider]", "policy.d=32", "policy.hidden=64", "encoder.dim=32", "algo.lr_actor=0.003",
            "algo.batch_size=64", "train.questions_per_iter=4", "train.eval_every=20", "train.eval_on=train",
            "train.curriculum_iters=0", "reward.variant=R1", "train.iterations=40"]
    assert train.main(args) == 0
    run = tmp_path / "runs" / "g"
    result = json.loads((run / "result.json").read_text())
    assert result["episodes_per_iteration"] == 32 and result["algo"] == "grpo"
    assert result["train_pool_greedy"]["em"] == 1.0
    rows = [json.loads(x) for x in (run / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 40 and "zero_adv_frac" in rows[0]


def test_candidates_are_shared_inside_a_group(setup):
    env, feat, policy, golds = setup
    calls = []
    real = env.candidate_actions
    env.candidate_actions = lambda s: (calls.append(s.t), real(s))[1]
    cache = {}
    collect(env, PolicyActor(policy, feat), MockReader(), [_inst()], mode="sample", oracle_mode="reward",
            golds=golds, group_size=8, generator=torch.Generator().manual_seed(0), cand_cache=cache)
    assert calls.count(0) == 1 and len(calls) == len(cache)  # one enumeration per distinct state
    n = len(calls)
    collect(env, PolicyActor(policy, feat), MockReader(), [_inst()], mode="argmax", oracle_mode="metric_only",
            golds=golds, cand_cache=cache)
    assert len(calls) <= n + 4 and calls.count(0) == 1  # a later call reuses them


def test_train_entry_point_ppo_overfits_one_question(tmp_path, data_dir, gold):  # noqa: F811
    """Gate G-8 in miniature: PPO from a random start, sparse reward, mock reader."""
    import json

    from mhtable import train

    _write_repo(data_dir, gold)
    cfg = Path(__file__).parents[1] / "configs" / "experiments" / "ppo.yaml"
    args = ["--config", str(cfg), "--data-dir", str(data_dir), "--out", str(tmp_path / "runs"), "--run-id", "p",
            "data.sources=[spider]", "policy.d=32", "policy.hidden=64", "encoder.dim=32", "algo.lr_actor=0.003",
            "algo.lr_critic=0.003", "algo.batch_size=64", "train.questions_per_iter=32", "train.eval_every=20",
            "train.eval_on=train", "train.curriculum_iters=0", "reward.variant=R1"]
    assert train.main(args + ["train.iterations=40"]) == 0
    run = tmp_path / "runs" / "p"
    result = json.loads((run / "result.json").read_text())
    assert result["train_pool_gold_path"]["em"] == 1.0
    assert result["train_pool_greedy"]["em"] == 1.0 and result["eval_best"]["eval"]["em"] == 1.0
    assert result["train_last10_sampled"]["em"] > 0.8
    assert len((run / "train.jsonl").read_text().splitlines()) == 40 and (run / "last.pt").exists()
    assert train.main(args + ["train.iterations=42", "--resume"]) == 0  # continues from last.pt
    assert len((run / "train.jsonl").read_text().splitlines()) == 42
