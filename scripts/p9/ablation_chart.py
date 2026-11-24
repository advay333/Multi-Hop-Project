"""Forest plot: every RL run's test EM difference vs BC (paired, 95% CI), strict and relaxed."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path("runs/hf/prajna")
rng = np.random.default_rng(0)
INK, MUTED, GRID, BG = "#1B2433", "#5A6475", "#D9DDE3", "#F7F5F0"
BLUE, ORANGE = "#2B5FAD", "#D9731A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False, "figure.facecolor": BG,
                     "axes.facecolor": BG, "legend.frameon": False})


def em(run, tier):
    f = R / f"t-{run}-test" / f"predictions_{tier}.jsonl"
    if not f.exists():
        return None
    return {json.loads(l)["instance_id"]: json.loads(l)["em"] for l in f.open(encoding="utf-8")}


bc = {t: em(f"bc-{t}", t) for t in ("strict", "relaxed")}


def delta(run, tier):
    p = em(run, tier)
    if p is None:
        return None
    ids = sorted(p)
    d = np.array([p[i] - bc[tier][i] for i in ids]) * 100
    m = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)  # same draws for every run
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


groups = [
    ("From BC · main grid (lr 1e-5)", [("PPO R1", "ppo-r1-{t}-lr1e-5-s0"), ("PPO R0", "ppo-r0-{t}-lr1e-5-s0"),
                                       ("GRPO R1", "grpo-r1-{t}-lr1e-5-s0"), ("GRPO R0", "grpo-r0-{t}-lr1e-5-s0")]),
    ("From BC · process GRPO (R1)", [("λ = 0.5", "grpoproc-r1-{t}-lr1e-5-lam0.5-s0"),
                                     ("λ = 0.95", "grpoproc-r1-{t}-lr1e-5-lam0.95-s0")]),
    ("From BC · bigger steps (R1, lr 1e-4)", [("PPO", "ppo-r1-{t}-big-s0"), ("GRPO", "grpo-r1-{t}-big-s0"),
                                              ("PPO + entropy", "ppo-r1-{t}-bigx-s0"),
                                              ("GRPO + entropy", "grpo-r1-{t}-bigx-s0")]),
    ("From BC · coverage reward (R1, lr 1e-4)", [("PPO, c = 0.5", "ppo-r1-{t}-cov0.5-s0"),
                                                 ("GRPO, c = 0.5", "grpo-r1-{t}-cov0.5-s0"),
                                                 ("PPO, c = 1", "ppo-r1-{t}-cov1-s0"),
                                                 ("GRPO, c = 1", "grpo-r1-{t}-cov1-s0")]),
    ("From scratch (no BC)", [("PPO R1", "ppo-r1-{t}-scratch-s0"), ("PPO R0", "ppo-r0-{t}-scratch-s0"),
                              ("GRPO R1", "grpo-r1-{t}-scratch-s0")]),
]

rows, heads = [], []
y = 0
for g, runs in groups:
    heads.append((y, g))
    y += 1
    for lab, pat in runs:
        rows.append((y, lab, delta(pat.format(t="strict"), "strict"), delta(pat.format(t="relaxed"), "relaxed")))
        y += 1
    y += 0.4

rows_last = rows[-1][0]
fig, ax = plt.subplots(figsize=(16, 10.2), dpi=100)
ax.axvline(0, color=INK, lw=2)
for yy, lab, s, r in rows:
    for d, c, off in ((s, BLUE, -0.17), (r, ORANGE, 0.17)):
        if d is None:
            ax.text(3.4, yy + off, "relaxed: not run", color=MUTED, fontsize=14, va="center")
            continue
        m, lo, hi = d
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [yy + off] * 2, color=c, lw=3, solid_capstyle="round")
        ax.plot(m, yy + off, "o", color=c, ms=12, mfc=c if sig else BG, mew=3)
for yy, g in heads:
    ax.text(-9.6, yy, g, fontsize=18, fontweight="bold", color=INK, va="center")
ax.set_yticks([r[0] for r in rows], [r[1] for r in rows], fontsize=18)
ax.set_ylim(y - 0.2, -0.8)
ax.set_xlim(-9.6, 5.6)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=1)
ax.set_xlabel("Test EM difference vs BC, percentage points (95% CI)", color=INK)
ax.text(0.15, -0.55, "better than BC →", color=MUTED, fontsize=15)
ax.text(-0.15, -0.55, "← worse than BC", color=MUTED, fontsize=15, ha="right")
ax.plot([], [], "o", color=BLUE, ms=12, mew=3, label="strict")
ax.plot([], [], "o", color=ORANGE, ms=12, mew=3, label="relaxed")
ax.plot([], [], "o", color=MUTED, mfc=BG, ms=12, mew=3, label="hollow = not significant")
ax.legend(loc="upper left", bbox_to_anchor=(0.3, 0.86), fontsize=16)
fig.tight_layout()
fig.savefig(Path(sys.argv[1]) / "ablation.png")
print("ok")
