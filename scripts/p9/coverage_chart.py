"""Coverage-reward chart for the deck: test EM difference vs BC (paired, 95% CI) by coverage weight c,
PPO vs GRPO, one panel per tier. c = 0 is the matching big-step run (same settings, EM-only reward).

    PYTHONPATH=<matplotlib dir> .venv/Scripts/python.exe scripts/p9/coverage_chart.py <out dir>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path("runs/hf/prajna")
INK, MUTED, GRID, BG = "#1B2433", "#5A6475", "#D9DDE3", "#F7F5F0"
COLORS = {"ppo": "#2B5FAD", "grpo": "#D9731A"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 19, "axes.edgecolor": MUTED,
                     "xtick.color": INK, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG,
                     "legend.frameon": False})


def em(run, tier):
    f = R / f"t-{run}-test" / f"predictions_{tier}.jsonl"
    return {json.loads(l)["instance_id"]: json.loads(l)["em"] for l in f.open(encoding="utf-8")}


def delta(run, tier, bc):
    p = em(run, tier)
    ids = sorted(p)
    d = np.array([p[i] - bc[i] for i in ids]) * 100
    m = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


RUN = {"0": "{a}-r1-{t}-big-s0", "0.5": "{a}-r1-{t}-cov0.5-s0", "1": "{a}-r1-{t}-cov1-s0"}
fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), dpi=100, sharey=True)
for ax, tier in zip(axes, ("strict", "relaxed")):
    bc = em(f"bc-{tier}", tier)
    ax.axhline(0, color=INK, lw=2)
    for a, off in (("ppo", -0.09), ("grpo", 0.09)):
        xs, ms = [], []
        for k, (c, pat) in enumerate(RUN.items()):
            m, lo, hi = delta(pat.format(a=a, t=tier), tier, bc)
            sig = lo > 0 or hi < 0
            x = k + off
            ax.plot([x, x], [lo, hi], color=COLORS[a], lw=3.5, solid_capstyle="round")
            ax.plot(x, m, "o", ms=15, color=COLORS[a], mfc=COLORS[a] if sig else BG, mew=3.5)
            ax.text(x + (0.07 if a == "grpo" else -0.07), m, f"{m:+.1f}", fontsize=16, color=COLORS[a],
                    ha="left" if a == "grpo" else "right", va="center")
            xs.append(x)
            ms.append(m)
        ax.plot(xs, ms, color=COLORS[a], lw=1.5, alpha=0.5)
    ax.set_xticks(range(3), ["c = 0\n(EM only)", "c = 0.5", "c = 1\n(rows only)"], fontsize=18)
    ax.set_xlim(-0.5, 2.5)
    ax.set_title(f"{tier.capitalize()} tier (BC = {np.mean(list(bc.values())) * 100:.1f})", loc="left",
                 color=INK, fontsize=21, pad=12)
    ax.grid(axis="y", color=GRID)
axes[0].set_ylabel("Test EM − BC (points, 95% CI)")
axes[0].set_ylim(-3, 5.5)
axes[0].plot([], [], "o", ms=13, color=COLORS["ppo"], label="PPO")
axes[0].plot([], [], "o", ms=13, color=COLORS["grpo"], label="GRPO")
axes[0].plot([], [], "o", ms=13, color=MUTED, mfc=BG, mew=3, label="hollow = not significant")
axes[0].legend(loc="upper left", fontsize=16)
fig.tight_layout(w_pad=3)
fig.savefig(Path(sys.argv[1]) / "coverage.png")
print("ok")
