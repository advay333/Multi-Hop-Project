# P9 analysis scripts

Laptop-side analysis of the Prajna runs. Run from the project root after downloading results to
`runs/hf/prajna/` (gitignored) with `fetch_runs.py`. All numbers in PLAN.md P9 come from these.

| Script | What it prints / writes |
|---|---|
| `fetch_runs.py [substr ...]` | downloads results (result.json, metrics.jsonl, train.jsonl, config, test predictions) — no `.pt`, no reader caches |
| `all_runs.py` | one Markdown row per run and tier: test EM, paired Δ vs BC [95% CI], same sub-table as BC, path match, tokens, dev EM start → best → final, KL to BC |
| `analyse_cov.py` | coverage-reward grid vs BC and vs the matching big-step run (cov 0), plus training signals |
| `charts.py <out dir>` | the deck's PNG charts: BC curves, RL dev curves, RL training signals, test EM bars |
| `coverage_chart.py <out dir>` | coverage-reward chart (Δ vs BC by c, PPO vs GRPO, per tier) |
| `ablation_chart.py <out dir>` | forest plot of every RL run's test EM difference vs BC (deck appendix) |

CIs: paired bootstrap over test questions (4,000–5,000 resamples, percentile 2.5 / 97.5, rng seed 0);
they cover question sampling only, not training seeds (one seed per cell).

The chart scripts need matplotlib, which is not in `.venv`; install it to a side folder and put it on
`PYTHONPATH`, e.g. `pip install --target <dir> matplotlib`, then
`PYTHONPATH=<dir> .venv/Scripts/python.exe scripts/p9/charts.py <out dir>`.
