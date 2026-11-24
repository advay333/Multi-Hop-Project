"""Slide charts from runs/hf/prajna (run from the project root)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path("runs/hf/prajna")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID, BG = "#1B2433", "#5A6475", "#D9DDE3", "#F7F5F0"
BLUE, ORANGE, TEAL, GREY = "#2B5FAD", "#D9731A", "#2E8B7A", "#9AA3AF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 19, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG,
                     "axes.facecolor": BG, "legend.frameon": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 1})


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def dev_curve(run):
    rows = {}
    for r in jl(R / run / "metrics.jsonl"):
        if r.get("eval"):
            rows[r["iteration"]] = r["eval"]["em"] * 100  # last row per iteration wins (resumes)
    it = sorted(rows)
    return np.array(it), np.array([rows[i] for i in it])


def smooth(y, w):
    y = np.asarray(y, float)
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


# 1. BC: training loss falls; join-path accuracy on dev rises -------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=100)
for tier, c in (("strict", BLUE), ("relaxed", ORANGE)):
    m = jl(R / f"bc-bge-{tier}-s0" / "metrics.jsonl")
    ep = [r["epoch"] for r in m]
    a1.plot(ep, [r["train"]["loss"] for r in m], color=c, lw=3.5, marker="o", ms=8, label=f"train, {tier}")
    a1.plot(ep, [r["dev_teacher_forced"]["loss"] for r in m], color=c, lw=2.5, ls="--", marker="o", ms=6,
            label=f"dev, {tier}")
    a2.plot(ep, [r["dev"]["path_match"] * 100 for r in m], color=c, lw=3.5, marker="o", ms=8, label=tier)
a1.set_title("Cross-entropy loss", loc="left", color=INK, fontsize=21, pad=12)
a1.set_xlabel("epoch")
a1.legend(fontsize=16, ncol=2, loc="upper center")
a2.set_title("Dev join-path accuracy (%)", loc="left", color=INK, fontsize=21, pad=12)
a2.set_xlabel("epoch")
a2.set_ylim(25, 50)
a2.legend(fontsize=16)
fig.tight_layout(w_pad=4)
fig.savefig(OUT / "bc_curves.png")
plt.close(fig)

# 2. RL learning curves: from scratch learns, from BC stays flat ------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=100, sharey=True)
for run, lab, c in (("ppo-r1-strict-scratch-s0", "PPO R1", BLUE), ("ppo-r0-strict-scratch-s0", "PPO R0", GREY),
                    ("grpo-r1-strict-scratch-s0", "GRPO R1", ORANGE)):
    it, em = dev_curve(run)
    a1.plot(it, em, color=c, lw=3.2, label=lab)
a1.axhline(32.0, color=TEAL, lw=2.5, ls="--")
a1.text(1000, 33.2, "BC (start of the right panel)", color=TEAL, fontsize=16, ha="right")
a1.set_title("From a random policy (strict)", loc="left", color=INK, fontsize=21, pad=12)
a1.set_xlabel("RL iteration (128 episodes each)")
a1.set_ylabel("dev EM (%)")
a1.set_ylim(0, 40)
a1.legend(fontsize=16, loc="lower right")
for run, lab, c, ls in (("ppo-r1-strict-lr1e-5-s0", "PPO R1, lr 1e-5", BLUE, "-"),
                        ("grpo-r1-strict-lr1e-5-s0", "GRPO R1, lr 1e-5", ORANGE, "-"),
                        ("ppo-r1-strict-big-s0", "PPO R1, lr 1e-4", BLUE, ":"),
                        ("grpo-r1-strict-big-s0", "GRPO R1, lr 1e-4", ORANGE, ":")):
    it, em = dev_curve(run)
    a2.plot(it, em, color=c, lw=3.2, ls=ls, label=lab)
a2.set_title("From the BC policy (strict)", loc="left", color=INK, fontsize=21, pad=12)
a2.set_xlabel("RL iteration (128 episodes each)")
a2.legend(fontsize=16, loc="lower right", ncol=2)
fig.tight_layout(w_pad=3)
fig.savefig(OUT / "rl_curves.png")
plt.close(fig)

# 3. RL training signals: return rises, critic learns (from scratch) -------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=100)
for run, lab, c in (("ppo-r1-strict-scratch-s0", "PPO R1", BLUE), ("grpo-r1-strict-scratch-s0", "GRPO R1", ORANGE),
                    ("ppo-r0-strict-scratch-s0", "PPO R0", GREY)):
    tr = jl(R / run / "train.jsonl")
    it = [t["iteration"] for t in tr]
    a1.plot(it, smooth([t["em"] * 100 for t in tr], 25), color=c, lw=3, label=lab)
    if "value_loss" in tr[0]:
        a2.plot(it, smooth([t["value_loss"] for t in tr], 25), color=c, lw=3, label=lab)
a1.set_title("Training-episode EM (%), 25-iter mean", loc="left", color=INK, fontsize=21, pad=12)
a1.set_xlabel("RL iteration")
a1.legend(fontsize=16, loc="upper left")
a2.set_title("PPO critic (value) loss", loc="left", color=INK, fontsize=21, pad=12)
a2.set_xlabel("RL iteration")
a2.legend(fontsize=16)
fig.tight_layout(w_pad=4)
fig.savefig(OUT / "rl_signals.png")
plt.close(fig)

# 4. Test EM, strict vs relaxed --------------------------------------------------------------
def em(run, tier):
    p = R / f"t-{run}-test" / "result.json"
    return json.load(open(p))["tiers"][tier]["em"] * 100


rows = [("Random rows (B2)", "random_rows", None), ("Truncate tables (B1)", "truncate", None),
        ("Heuristic", "heuristic", None), ("BC", "bc-{t}", None),
        ("PPO R1", "ppo-r1-{t}-lr1e-5-s0", None), ("PPO R0", "ppo-r0-{t}-lr1e-5-s0", None),
        ("GRPO R1", "grpo-r1-{t}-lr1e-5-s0", None), ("GRPO R0", "grpo-r0-{t}-lr1e-5-s0", None),
        ("Oracle path", "oracle", None)]
labels, s, r = [], [], []
for lab, pat, _ in rows:
    labels.append(lab)
    s.append(em(pat.format(t="strict"), "strict"))
    r.append(em(pat.format(t="relaxed"), "relaxed"))
y = np.arange(len(rows))[::-1]
fig, ax = plt.subplots(figsize=(16, 7.4), dpi=100)
ax.barh(y + 0.2, s, 0.38, color=BLUE, label="strict")
ax.barh(y - 0.2, r, 0.38, color=ORANGE, label="relaxed")
for yy, a, b in zip(y, s, r):
    ax.text(a + 0.5, yy + 0.2, f"{a:.1f}", va="center", fontsize=16, color=INK)
    ax.text(b + 0.5, yy - 0.2, f"{b:.1f}", va="center", fontsize=16, color=INK)
ax.axhspan(y[7] - 0.5, y[4] + 0.5, color="#E6ECF5", zorder=0)
ax.set_yticks(y, labels, fontsize=19, color=INK)
ax.set_xlabel("Test exact-match accuracy (%), test set: 1,172 strict / 1,361 relaxed questions")
ax.set_xlim(0, 52)
ax.grid(axis="y", visible=False)
ax.legend(fontsize=17, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "test_em.png")
plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))
