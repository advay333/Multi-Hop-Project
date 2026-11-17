# CLAUDE.md — start here

Seminar project (IIT Bombay, Advay Bapat + Arijit Paul): **learning multi-hop table reasoning
with RL**. A small learned policy picks filters and joins over a relational database, SQLite
executes them, and a frozen LLM reader answers from the resulting sub-table. Training: behaviour
cloning, then PPO and GRPO; rewards = answer correctness + potential-based shaping from gold SQL
− token cost. Research question: does a learned critic (PPO) pay off over critic-free GRPO in this
short-horizon, small-action regime?

## Documents (read in this order)

| File | Role |
|---|---|
| `Multi-Hop_Table_RL_Project_Report.pdf` | Research design: problem, MDP, rewards, metrics, algorithm arms (Parts A–E) |
| `PLAN.md` | **Source of truth for status**: phases P0–P10 with gates, §4a pending items, §6 engineering rules |
| `DESIGN.md` | Architecture: storage, data model, every module's inputs/outputs, invariants, known data caveats (§11) |
| `data/DATASET_CARD.md` | Data provenance and every measured number (yields, oracle pass rates, coverage) |

`PROJECT.md` and `TASK_01_DATASET.md` were deleted on purpose (superseded); do not resurrect them.

## Status (2026-11-08)

- **P0 repo + storage — done.** **P1 data — done** (`data-v1`). **P2 oracle — done** for
  Spider+MMQA (`data-v2`). **P3 environment — done** for Spider+MMQA (gate G-6: train strict
  89.6%, relaxed 85.1%).
- **BIRD work (2026-11-16, PLAN §4a):** the 4 oracle fixes and an action-set revision are in the
  code (tests pass), not yet in the data. BIRD coverage measured offline: train strict 88.6%
  (passes) / relaxed 79.4%. The oracle gate (95%) is projected at ~93.4% relaxed because of
  genuine fan-out joins; that is the user's call after the rebuild.
- **Next step: `data-v3` rebuild on molab** (both families, then both indexes, then verify, then
  tag), after the user commits and pushes: `notebooks/rebuild_v3.py` does all of it (tag is a
  separate button). Steps listed in PLAN §4a.
- **Compute:** molab is the main compute **for now** — Prajna login is blocked on a two-factor
  reset (the QR code was never shown; needs a reset from hpc@iitb.ac.in). Prajna becomes main once
  access works (section below).
- **Next phase: P4** — readers (mock / HF / vLLM behind one interface, cached), encoders +
  embedding cache, metrics. Then P5 shared rollout + rewards + MLP policy + algorithm interface,
  P6 baselines + oracle ceiling, P7 BC, P8 PPO, P9 GRPO and the PPO/GRPO × sparse/shaped grid.
- Always track **strict and relaxed action tiers side by side** (user's request).

## Where things live

- Code: `src/mhtable/` (installable: `pip install -e ".[dev]"`), tests in `tests/` (`pytest`, ~5 s).
  Key modules: `data/` (norm, schema, sqlshape, execute, loaders, mmqa_db, split, build),
  `oracle.py`, `sqlrender.py`, `env/` (text, index, actions, env), `analysis/` (action_fit,
  coverage, reach — G-6 without databases, works for BIRD on the laptop), `storage.py`, `types.py`, `config.py`, `registry.py`.
- Data: **all on private HuggingFace repos** under `advay333/`: `mhtable-raw` (sources),
  `mhtable-data` (built data, SQLite DBs, `index/`; tags `data-v1`, `data-v2`), `mhtable-cache`,
  `mhtable-runs`. Local `data/` is only a cache (gitignored, except `data/DATASET_CARD.md`).
  Access via `mhtable.storage` (`fetch`, `push`, `push_large`, `db_path`).
- **Compute: molab now; Prajna HPC once login works** (see the section below). **molab** (marimo notebooks in `notebooks/`, which only clone the GitHub repo
  `advay333/Multi-Hop-Project`, install it and run one `python -m mhtable...` command):
  `source_data.py`, `build_data.py` (`--family spider_mmqa|bird`), `build_index.py`,
  `rebuild_v3.py` (whole data-v3 rebuild + verification + tag).
- Laptop: Windows, RTX 4060 8 GB, slow network (~370 KB/s) — development and unit tests only;
  never download BIRD databases (32 GB) locally.

## Prajna HPC (main compute once login works; molab until then)

Source: `Updated User Manual for Prajna (AiML) HPC.pdf` in the project root. Re-read it when in
doubt; this section summarises it and our rules on top of it.

**What it is.** IIT Bombay's Prajna (AI-ML) cluster: Rocky Linux 9, x86_64, SLURM 23.11,
Lustre parallel filesystem (2 PB), InfiniBand. GPU partitions (manual p. 10):

| Nodes | GPUs per node | Partition name in manual |
|---|---|---|
| 9 DGX A100 | 8× A100 | not stated |
| 20 Exatron | 4× A40 (48 GB) — node example: 64 cores, 512 GB RAM | `a40` |
| 7 Exatron | 8× L40S (48 GB) | `l40` (shown as default) |
| 10 Tyron | 2× L4 (24 GB) | not stated |

**Access** (course instructions for CS769, confirmed separately from the manual).
`ssh <LDAP-ID>@prajna.iitb.ac.in` from PowerShell/terminal (or PuTTY/MobaXterm, or VS Code
Remote-SSH with a `Host prajna.iitb.ac.in` entry). Login = password + a 6-digit two-factor code
from Google Authenticator (the QR code is shown once, on the first ssh). Change the initial
password with `passwd` right after the first login. File transfer:
`scp -r <local> <LDAP-ID>@prajna.iitb.ac.in:<remote dir>` (the manual's example uses
`10.195.100.101`). Support: https://help.cc.iitb.ac.in/ or hpc@iitb.ac.in. Never store passwords
or 2FA codes in the repo.

**Filesystems.**
- `/home/<groupname>/<username>/` — persistent, quota-limited. Code checkout, Python env, small
  configs.
- `/scratch` — temporary working data. **Files not accessed for 3 months are deleted.** Use it
  for HF caches (`HF_HOME`, `MHTABLE_CACHE_DIR`, `MHTABLE_WORK_DIR`), extracted data, and run
  outputs — and push anything worth keeping to HuggingFace (`mhtable-runs`, `mhtable-cache`).

**Rules (manual + ours).**
1. **Never run compute on the login node** — it is shared and such jobs are killed. Login node
   is for editing, `git pull`, file transfer, environment setup and submitting jobs.
2. **Everything heavy goes through SLURM** `sbatch` scripts kept in the repo under `slurm/`
   (version-controlled, like the notebooks: no logic in them — they activate the env, `cd` to
   the checkout and run one `python -m mhtable...` command). Each script sets `--job-name`,
   `--partition`, `--qos` (if required), `--gres=gpu:N`, `--time`, `--output`/`--error`
   (`%J` = job id, `%A_%a` for arrays).
3. Ask for **realistic `--time` limits**: Prajna uses backfill scheduling, so accurate limits get
   jobs started sooner. Checkpoint to `/scratch` and HF so jobs can resume.
4. Use **job arrays** (`--array=0-29%4`) for seeds × arms grids and **dependencies**
   (`-d afterok:<id>` / `-d singleton`) for build → index → train pipelines.
5. Request only the GPUs a job uses; RL policy training is CPU-light — the GPU is for the reader
   (vLLM) and the encoder.
6. **The user submits, monitors and cancels jobs** (`sbatch`, `squeue --me`, `scontrol show job`,
   `scancel`, `sacct`); agents write and explain the scripts. Same commit rule as everywhere: the
   user commits and pushes.
7. Software: Spack (`source /lustre-flash/apps/spack/share/spack/setup-env.sh`, `spack find`,
   `spack load <pkg@ver>`); project Python env lives in `/home` (venv or conda), installed once
   with `pip install -e ".[dev]"`.
8. Back up results: anything only on `/scratch` is at risk after 3 months.

**Missing information — confirm on first login and record here** (not in the manual):
- Exact partition and QoS names for the A100 and L4 nodes, our account's allowed partitions,
  walltime and job/GPU limits — manual points to https://hpcverse.iitb.ac.in/queue-policy/prajna
  and `sinfo` / `scontrol show partition <name>`.
- Whether **compute nodes have internet access** (needed for HF downloads and pip). If not:
  fetch data/models on the login node into `/scratch`, then run jobs with `HF_HUB_OFFLINE=1`.
- `/home` and `/scratch` quotas for our group (data repo is ~33 GB; model weights ~15 GB).
- Available Python/CUDA versions via Spack, and whether vLLM installs cleanly (driver version).
- Our SLURM account name (`--account`) if more than one is assigned.
- Whether interactive GPU sessions (`salloc`/`srun --pty`) are allowed for debugging.

## Rules for agents

- **Never `git commit` or `git push`.** The user does all commits; say when it is a good point to
  commit, list the files, and suggest a message.
- Never upload to HuggingFace from the laptop without asking; uploads normally run in Prajna jobs or molab
  notebooks the user starts.
- Keep `PLAN.md`, `DESIGN.md` and `data/DATASET_CARD.md` current when anything changes.
- The user has no prior SQL background: explain database concepts plainly.
- Firewall: gold SQL / answers / entity sets reach only the oracle and rewards — never the
  environment, featuriser or reader prompt (`tests/test_env.py` enforces this for `env/`).
- Shell quirks on this machine: use the Write tool for multi-line Python edits — bash heredocs
  sometimes break on quotes, and `\b` in edit scripts became backspace characters once.
