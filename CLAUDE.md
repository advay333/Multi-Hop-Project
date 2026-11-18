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

## Status (2026-11-17)

- **P0 repo + storage — done.** **P1 data — done** (`data-v1`). **P2 oracle — done.**
  **P3 environment — done.** Current data: **`data-v3`** (tagged 2026-11-17 on mhtable-data; data +
  index together; use `data.revision: data-v3`).
- **data-v3 pool** (oracle-verified, strict / relaxed): train Spider 1,707 / 1,944, MMQA 942 /
  1,219, BIRD 3,038 / 4,679; dev 394 / 458; test 1,172 / 1,361 (Spider+MMQA only); bird_dev
  643 / 832 (transfer set). G-6 reachability: Spider+MMQA train 91.0% / 86.2% (pass), BIRD train
  87.4% / 79.5%. Fully reachable ≈ 5,060 strict / 6,450 relaxed (DATASET_CARD "data-v3").
- **User decisions 2026-11-17:** accept the BIRD oracle pass rate (~93%, gate 95%) and BIRD
  relaxed reachability 79.5% (gate 85%). Spider+MMQA passes every gate.
- **Open (PLAN §4a): BIRD real-SQL check did not finish** — look-ahead COUNTs on the largest
  BIRD DBs (`movie_platform`, `talkingdata`) are far too slow (90+ min for part of one tier), so
  crash-freedom on real BIRD DBs is unverified and RL on BIRD will be slow until fixed (SQLite
  indexes on join/anchor columns → `data-v4`, a row-count cap, or excluding the largest DBs).
  Train on Spider+MMQA first. The data-v3 per-tier oracle `fail_reasons` were not seen (molab
  session lost); `stats/bird.json` on HF has them if the build pushed `stats/`.
- **Compute:** molab is the main compute **for now** — Prajna login is blocked on a two-factor
  reset (the QR code was never shown; needs a reset from hpc@iitb.ac.in). Prajna becomes main once
  access works (section below).
- **P5 learning core — done 2026-11-17** (built before P4 because it needs no molab run; 93 tests
  pass): `featurise.py`, `policies/mlp.py`, `rewards.py`, `rollout.py`, `algos/base.py`,
  `firewall.py`, `sufficiency.py`, plus the reader/encoder interface slices (`readers/base.py`,
  `prompt.py`, `mock.py`; `encoders/base.py` with a `hash` test encoder, `cache.py`; `metrics.py`
  with EM/set-F1 only). Needs `torch` (CPU build installed in `.venv`; extra `train`).
- **P7 behaviour cloning — code done 2026-11-17** (95 tests pass): `dataset.py`, `algos/bc.py`,
  `train.py`, `configs/experiments/bc.yaml`; runs land in local `runs/` (gitignored). First runs
  with the `hash` encoder + mock reader on data-v2 Spider+MMQA: dev join-path accuracy strict
  31.5% / relaxed 25.3% (untrained 0.5%), train subset 77% / 72% — large generalisation gap,
  expected to narrow with real embeddings (PLAN P7). Gate (beat the P6 heuristic) still open.
- **P8 PPO — code done 2026-11-17** (98 tests pass): `algos/ppo.py`, RL loop in `train.py`
  (curriculum, eval, best/last checkpoints, `--resume`), `selectors/oracle.py`, candidate memo,
  `train.device` (installed torch is CPU-only, so the RTX 4060 is unused). Runs on data-v2
  Spider+MMQA, hash encoder + mock reader. **Gate G-8 (overfit 20 questions) not passed:** R1
  80% / R0 55% vs ceiling 100% — entropy collapses early. From the BC checkpoint, dev EM 37% →
  ~48% (R1 and R0 indistinguishable, one seed), mostly by taking larger sub-tables (recorded in
  PLAN P8).
- **P4 code — done 2026-11-18** (107 tests + 2 `network` tests; PLAN P4): `vllm` / `hf` readers
  (`readers/llm.py` shared), reader tokenizer in `materialise` (`env.tokenizer: reader`), bge-m3
  `encoders/st.py` + `encoders/build_cache.py`, metric suite, `evaluate.py`,
  `configs/experiments/eval.yaml`, molab notebook `notebooks/p4_reader.py`. Gate = that
  notebook's run (not yet done; also builds the bge-m3 cache for spider_mmqa and bird).
  **Projection fixed 2026-11-18:** `materialise` now shows every column of the visited tables
  that fits the budget (`env.project_all_columns`); answer_in_subtable on the gold path 69% → 91%
  strict, 48% → 88% relaxed (PLAN P4).
- **Next (PLAN §4b, agreed 2026-11-18): three independent tracks, then P9.** A: fix G-8 (laptop,
  mock reader). B: BIRD look-ahead speed (molab only). C: P4 (hf / vLLM readers, bge-m3 encoder +
  embedding cache, full metric suite, reader tokenizer in `materialise`) + P6 baselines — the
  critical path. §4b has laptop vs molab vs Kaggle run-time estimates. Only one molab run at a time.
- **Kaggle (free, 2× T4) is a second GPU box** (PLAN §1, §4b): embedding cache, baselines,
  part of the P9 grid. Same reader weights/precision/vLLM version on every platform for any
  compared runs.
- Always track **strict and relaxed action tiers side by side** (user's request).

## Where things live

- Code: `src/mhtable/` (installable: `pip install -e ".[dev]"`), tests in `tests/` (`pytest`, ~30 s with torch).
  Key modules: `data/` (norm, schema, sqlshape, execute, loaders, mmqa_db, split, build),
  `oracle.py`, `sqlrender.py`, `env/` (text, index, actions, env), learning core (`featurise.py`,
  `policies/`, `rewards.py`, `rollout.py`, `algos/base.py`, `readers/`, `encoders/`, `metrics.py`,
  `firewall.py`, `sufficiency.py`), `analysis/` (action_fit,
  coverage, reach — G-6 without databases, works for BIRD on the laptop), `storage.py`, `types.py`, `config.py`, `registry.py`.
- Data: **all on private HuggingFace repos** under `advay333/`: `mhtable-raw` (sources),
  `mhtable-data` (built data, SQLite DBs, `index/`; tags `data-v1`, `data-v2`), `mhtable-cache`,
  `mhtable-runs`. Local `data/` is only a cache (gitignored, except `data/DATASET_CARD.md`).
  Access via `mhtable.storage` (`fetch`, `push`, `push_large`, `db_path`).
- **Compute: molab now; Prajna HPC once login works** (see the section below). **molab** (marimo notebooks in `notebooks/`, which only clone the GitHub repo
  `advay333/Multi-Hop-Project`, install it and run one `python -m mhtable...` command):
  `source_data.py`, `build_data.py` (`--family spider_mmqa|bird`), `build_index.py`,
  `rebuild_v3.py` (whole data-v3 rebuild + verification + tag), `p4_reader.py` (vLLM reader
  smoke test, bge-m3 embedding cache, P4 gate).
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
