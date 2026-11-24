"""Coverage-reward grid vs BC and vs the matching big-step run (cov 0)."""
import hashlib
import json
from pathlib import Path

import numpy as np

R = Path("runs/hf/prajna")
rng = np.random.default_rng(0)


def preds(run, tier):
    f = R / f"t-{run}-test" / f"predictions_{tier}.jsonl"
    out = {}
    for line in f.open(encoding="utf-8"):
        d = json.loads(line)
        out[d["instance_id"]] = dict(em=d["em"], sub=hashlib.md5((d.get("subtable") or "").encode()).hexdigest(),
                                     tok=d.get("tokens") or 0, path=d.get("path_match") or 0,
                                     suff=d.get("sufficient") or 0, ais=d.get("answer_in_subtable") or 0)
    return out


def boot(d):
    m = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
    return f"{d.mean()*100:+.1f} [{np.percentile(m,2.5)*100:+.1f}, {np.percentile(m,97.5)*100:+.1f}]"


def dev(run):
    rows = {}
    for line in (R / run / "metrics.jsonl").open():
        r = json.loads(line)
        if r.get("eval"):
            rows[r["iteration"]] = r["eval"]
    it = sorted(rows)
    em = [rows[i]["em"] * 100 for i in it]
    return f"{em[0]:.1f} -> best {max(em):.1f} -> final {em[-1]:.1f}", rows[it[-1]].get("sufficient")


def train(run):
    tr = [json.loads(l) for l in (R / run / "train.jsonl").open()]
    g = lambda k: np.mean([t[k] for t in tr if t.get(k) is not None]) if any(t.get(k) is not None for t in tr) else float("nan")
    first, last = tr[:25], tr[-25:]
    return dict(suff_first=np.mean([t["sufficient"] for t in first]), suff_last=np.mean([t["sufficient"] for t in last]),
                em_first=np.mean([t["em"] for t in first]), em_last=np.mean([t["em"] for t in last]),
                zero=g("zero_adv_frac"), kl=tr[-1].get("kl_bc"), ent=tr[-1]["entropy"], calls=tr[-1]["reader_calls"])


for tier in ("strict", "relaxed"):
    bc = preds(f"bc-{tier}", tier)
    ids = sorted(bc)
    col = lambda p, k: np.array([p[i][k] for i in ids], float)
    print(f"\n===== {tier}: BC EM {col(bc,'em').mean()*100:.1f} suff {col(bc,'suff').mean()*100:.1f} "
          f"ans_shown {col(bc,'ais').mean()*100:.1f} path {col(bc,'path').mean()*100:.1f} tok {col(bc,'tok').mean():.0f}")
    for algo in ("ppo", "grpo"):
        base = f"{algo}-r1-{tier}-big-s0"
        pb = preds(base, tier)
        for run in (base, f"{algo}-r1-{tier}-cov0.5-s0", f"{algo}-r1-{tier}-cov1-s0"):
            p = preds(run, tier)
            same = np.mean([p[i]["sub"] == bc[i]["sub"] for i in ids]) * 100
            line = (f"{run:28s} EM {col(p,'em').mean()*100:5.1f} | vsBC {boot(col(p,'em')-col(bc,'em'))}"
                    + ("" if run == base else f" | vsBig {boot(col(p,'em')-col(pb,'em'))}")
                    + f" | suff {col(p,'suff').mean()*100:4.1f} ans_shown {col(p,'ais').mean()*100:4.1f} "
                      f"path {col(p,'path').mean()*100:4.1f} tok {col(p,'tok').mean():4.0f} sameBC {same:3.0f}%")
            print(line)
            d, dsuff = dev(run)
            t = train(run)
            print(f"    dev EM {d} (final dev suff {dsuff}) | train suff {t['suff_first']:.2f}->{t['suff_last']:.2f} "
                  f"em {t['em_first']:.2f}->{t['em_last']:.2f} | zero_adv {t['zero']:.2f} | KL_bc {t['kl']} ent {t['ent']:.2f} calls {t['calls']}")
