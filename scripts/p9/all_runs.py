"""One row per run: test EM, paired delta vs BC (95% CI), same sub-table as BC, path match, tokens, dev curve."""
import hashlib
import json
from pathlib import Path

import numpy as np

R = Path("runs/hf/prajna")
rng = np.random.default_rng(0)


def preds(run, tier):
    f = R / f"t-{run}-test" / f"predictions_{tier}.jsonl"
    if not f.exists():
        return None
    out = {}
    for line in f.open(encoding="utf-8"):
        d = json.loads(line)
        out[d["instance_id"]] = (d["em"], hashlib.md5((d.get("subtable") or "").encode()).hexdigest(),
                                 d.get("tokens") or 0, d.get("path_match") or 0)
    return out


def dev(run):
    f = R / run / "metrics.jsonl"
    if not f.exists():
        return ""
    rows = {}
    for line in f.open():
        r = json.loads(line)
        if r.get("eval"):
            rows[r["iteration"]] = r["eval"]["em"] * 100
    it = sorted(rows)
    if not it:
        return ""
    return f"{rows[it[0]]:.1f} → {max(rows.values()):.1f} → {rows[it[-1]]:.1f}"


def kl(run):
    f = R / run / "train.jsonl"
    if not f.exists():
        return ""
    last = json.loads(f.read_text().strip().splitlines()[-1])
    return f"{last.get('kl_bc', 0):.2f}" if "init" not in run and last.get("kl_bc") else "—"


bc = {t: preds(f"bc-{t}", t) for t in ("strict", "relaxed")}


def cell(run, tier):
    p = preds(run, tier)
    if p is None:
        return None
    ids = sorted(p)
    e = np.array([p[i][0] for i in ids])
    out = {"em": e.mean() * 100, "tok": np.mean([p[i][2] for i in ids]), "path": np.mean([p[i][3] for i in ids]) * 100}
    if not run.startswith("bc-") and run not in ("oracle", "heuristic", "truncate", "random_rows"):
        b = bc[tier]
        d = e - np.array([b[i][0] for i in ids])
        m = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
        out["d"] = (d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100)
        out["same"] = np.mean([p[i][1] == b[i][1] for i in ids]) * 100
    return out


groups = [
    ("Baselines", [("Random rows (B2)", "random_rows", None), ("Truncate tables (B1)", "truncate", None),
                   ("Heuristic", "heuristic", None), ("Oracle path (uses gold)", "oracle", None),
                   ("BC (bge-m3)", "bc-{t}", "bc-bge-{t}-s0")]),
    ("P9 gate (PPO R1 from BC)", [("PPO R1, lr 3e-4", None, "ppo-r1-{t}-lr3e-4-s0")]),
    ("Main grid (from BC, lr 1e-5)", [(f"{a} {r}", f"{a.lower()}-{r.lower()}-{{t}}-lr1e-5-s0", f"{a.lower()}-{r.lower()}-{{t}}-lr1e-5-s0")
                                      for a in ("PPO", "GRPO") for r in ("R1", "R0")]),
    ("Process GRPO (from BC, lr 1e-5)", [(f"Process GRPO R1, λ {l}", f"grpoproc-r1-{{t}}-lr1e-5-lam{l}-s0", f"grpoproc-r1-{{t}}-lr1e-5-lam{l}-s0")
                                         for l in ("0.5", "0.95")]),
    ("Bigger steps (from BC, R1)", [(f"{a} big", f"{a.lower()}-r1-{{t}}-big-s0", f"{a.lower()}-r1-{{t}}-big-s0") for a in ("PPO", "GRPO")]
     + [(f"{a} bigx", f"{a.lower()}-r1-{{t}}-bigx-s0", f"{a.lower()}-r1-{{t}}-bigx-s0") for a in ("PPO", "GRPO")]),
    ("Coverage reward (from BC, R1, lr 1e-4)", [(f"{a} c={c}", f"{a.lower()}-r1-{{t}}-cov{c}-s0", f"{a.lower()}-r1-{{t}}-cov{c}-s0")
                                                for c in ("0.5", "1") for a in ("PPO", "GRPO")]),
    ("From scratch (no BC)", [("PPO R1", "ppo-r1-{t}-scratch-s0", "ppo-r1-{t}-scratch-s0"),
                              ("PPO R0", "ppo-r0-{t}-scratch-s0", "ppo-r0-{t}-scratch-s0"),
                              ("GRPO R1", "grpo-r1-{t}-scratch-s0", "grpo-r1-{t}-scratch-s0")]),
]

print("| Group | Run | Tier | Test EM | Δ vs BC [95% CI] | Same sub-table as BC | Path match | Tokens | Dev EM start → best → final | KL to BC (end) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for g, runs in groups:
    for lab, test, train in runs:
        for t in ("strict", "relaxed"):
            c = cell(test.format(t=t), t) if test else None
            trn = train.format(t=t) if train else None
            if c is None and (trn is None or not (R / trn).exists()):
                print(f"| {g} | {lab} | {t} | not run / missing | | | | | | |")
                continue
            em = f"{c['em']:.1f}" if c else "— (no test run)"
            d = f"{c['d'][0]:+.1f} [{c['d'][1]:+.1f}, {c['d'][2]:+.1f}]" if c and "d" in c else ""
            same = f"{c['same']:.0f}%" if c and "same" in c else ""
            path = f"{c['path']:.1f}" if c else ""
            tok = f"{c['tok']:.0f}" if c else ""
            dv = dev(trn) if trn else ""
            k = kl(trn) if trn and "bc-" not in trn else ""
            print(f"| {g} | {lab} | {t} | {em} | {d} | {same} | {path} | {tok} | {dv} | {k} |")
