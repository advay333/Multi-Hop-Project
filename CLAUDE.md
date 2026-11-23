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
  **P3 environment — done.** Current data: **`data-v4`** (tagged 2026-11-21 on mhtable-data;
  = data-v3 of 2026-11-17 + SQLite indexes in the large BIRD DBs; use `data.revision: data-v4`).
- **data-v3 pool** (oracle-verified, strict / relaxed): train Spider 1,707 / 1,944, MMQA 942 /
  1,219, BIRD 3,038 / 4,679; dev 394 / 458; test 1,172 / 1,361 (Spider+MMQA only); bird_dev
  643 / 832 (transfer set). G-6 reachability: Spider+MMQA train 91.0% / 86.2% (pass), BIRD train
  87.4% / 79.5%. Fully reachable ≈ 5,060 strict / 6,450 relaxed (DATASET_CARD "data-v3").
- **User decisions 2026-11-17:** accept the BIRD oracle pass rate (~93%, gate 95%) and BIRD
  relaxed reachability 79.5% (gate 85%). Spider+MMQA passes every gate.
- **Closed by data-v4 (was open, PLAN §4a): BIRD real-SQL check did not finish** — look-ahead COUNTs on the largest
  BIRD DBs (`movie_platform`, `talkingdata`) are far too slow (90+ min for part of one tier), so
  crash-freedom on real BIRD DBs is unverified and RL on BIRD will be slow until fixed (SQLite
  indexes on join/anchor columns → `data-v4`, a row-count cap, or excluding the largest DBs).
  Train on Spider+MMQA first. The data-v3 per-tier oracle `fail_reasons` were not seen (molab
  session lost); `stats/bird.json` on HF has them if the build pushed `stats/`.
- **Track B — done 2026-11-21** (PLAN §4a; `notebooks/track_b.py`): **`data-v4`** = data-v3 +
  SQLite indexes + ANALYZE in 41 of 80 BIRD DBs (`env/sqlindex.py`); look-ahead memo + bounded
  `materialise` in `env/env.py`, `analysis/speed.py`, `coverage.py --per-db`,
  `storage download|push-large|tag` CLI (`push_large` works on huggingface_hub 2.x). BIRD real-SQL
  check: 0 crashes, reachability = offline G-6, look-ahead timeouts < 1% overall
  (language_corpus ~14%, accepted). **Use `data.revision: data-v4` everywhere** (Spider/MMQA
  files identical to v3). The BIRD "Open" item above is closed.
- **Compute:** Prajna access works since 2026-11-21 (account `cs769_02`; partitions and limits in
  the Prajna section below) and becomes the main compute once the `slurm/` scripts exist and
  the reader runs there (compute nodes have no internet: everything is fetched on the login
  node); molab remains the fallback.
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
- **Track C laptop work — done 2026-11-21** (111 tests): P6 baselines (`selectors/heuristic.py`,
  `TableEnv.materialise_tables` for B1 truncate / B2 random rows, `evaluate --selector
  heuristic|truncate|random_rows`), BC demonstrations in float16 (1024-d fits),
  `configs/experiments/bc_bge.yaml`, reader locked to Qwen2.5-7B + prompt p1 + bf16 in
  `eval.yaml`. **GPU run pending:** `notebooks/track_c.py` (BC with bge-m3, full-dev evaluation
  of oracle / B1 / B2 / heuristic / BC). Dry run: whole tables show the answer 93% of the time at
  ~700 tokens (dev DBs are small), so B1 may be a strong baseline (PLAN P6).
- **Track A / G-8 — closed 2026-11-22 (user accepted):**
  - From-scratch PPO plateaus at about 85% of the 20-question ceiling (laptop and Prajna sweeps, PLAN P8).
  - Cause: Adam steps at actor lr 3e-4 are about 100× too large for the 2.3M-parameter policy (one step moves a converged policy by KL 2.6).
  - New PPO options: `target_kl`, `freeze_temperature`, `adam_eps`.
- **GRPO code done 2026-11-22:** `algos/grpo.py`, `configs/experiments/grpo.yaml`; 124 tests pass.
- **Prajna:**
  - Done: `slurm/setup_env.sh`, `fetch_data.sh` and the `g8.sh` array; the user ran the G-8 sweep. Venv `~/mhtable-venv`, data `~/mhtable-data/v4`, runs `~/mhtable-runs`; submit from the repo root, logs in `slurm/logs/`.
  - Reader setup: first `reader_smoke` (2026-11-22) failed — the default vLLM wheel is a CUDA 13
    build; `setup_reader.sh` now installs the cu129 build (see "First batch job" below). Third
    smoke (cn43-l40): torch 2.11.0+cu129 + vllm 0.24.0 see the GPU and load Qwen2.5-7B offline
    from `~/.cache/huggingface` — the CUDA problem is solved — but vLLM's compile step then
    failed: Triton builds a small C helper with gcc, which needs Python's headers (`Python.h`;
    the system has no python3.11-devel). Fix: `setup_reader.sh` downloads standalone Python
    headers to `~/mhtable-pyinclude` (via `uv python install`), and **every GPU job sources
    `slurm/gpu_env.sh`** (venv, `HF_HUB_OFFLINE=1`, `C_INCLUDE_PATH`, a C build check).
  - **Reader works on Prajna (2026-11-22):** smoke on L40S, oracle + Qwen2.5-7B, 100 dev
    questions per tier: EM strict 59% / relaxed 53% (molab: 56.0% / 50.5%), 31k calls/h (PLAN P9).
    Next: BC with bge-m3 → PPO on dev with the real reader (the P9 gate). Scripts written:
    `slurm/bc_bge.sh` (array 0-1), then `slurm/ppo_gate.sh` (array 0-3,
    `configs/experiments/ppo_real.yaml`); both have `SMOKE=1`; not yet run. Gate = the in-run
    dev curve (final EM vs iteration 0 = BC), not the policy.pt re-score (that checkpoint
    includes iteration 0, so it cannot fall below BC). PLAN P9.
  - Smoke tests of `bc_bge.sh` and `ppo_gate.sh` passed 2026-11-22 (reader in the RL loop works;
    ~2–5 h per full RL run, estimated). Grid + test scripts written, not run: `slurm/p9_grid.sh`
    (array 0-7, needs `LR=`; `configs/experiments/grpo_real.yaml`), `slurm/baselines_test.sh`.
  - **P9 grid seeds (user decision 2026-11-22): seed 0 only for now** (8 cells, 2 of them = the
    gate's PPO-R1 runs); a second seed later if time permits. Single-seed caveat goes in the report.
  - **P9 gate DONE 2026-11-22.** Results:
    - In-run dev EM, BC start → final: strict lr 1e-5 32.0 → 32.0 (best 34.0); strict 3e-4 32.2 → 32.0, FAIL (crashed to 27, then recovered); relaxed 1e-5 27.5 → 30.3 (best 32.1); relaxed 3e-4 27.7 → 32.3.
    - Every gain is 1–2 SE (dev SE ≈ 2.3 pts). **Grid LR = 1e-5, PPO λ kept at 0.95** (user decision).
    - About 1 h per RL run on L40S/A40 (estimate was 2–5 h). A40 works too.
    - Read a gate with `python -m mhtable.analysis.gate <runs dir>`.
  - **Test baselines** (`baselines_test.sh`), EM strict / relaxed: oracle 46.4 / 44.8, BC 28.0 / 26.8, heuristic 20.5 / 20.0, B1 truncate 12.5 / 13.0, B2 random 10.6 / 10.9. B1 and B2 show the answer 80%+ of the time but whole tables confuse the reader. The test oracle is lower than dev's (59).
  - **Grid (`p9_grid.sh`, 8 cells) and process GRPO (`p9_process.sh`, R1 × 2 tiers × λ 0.5 / 0.95) both COMPLETED 2026-11-23; not yet analysed.**
    - Tasks 3 and 5 were resumed after a node fault on cn43-l40, so keep the last metrics row per iteration.
    - Results must be fetched: HF `mhtable-runs/prajna/` has only the BC runs (the first upload loop's globs ran in the wrong folder; corrected loop given), and the laptop `runs/prajna/` has only gate-era files.
    - Prajna paths: home `/home/cs769/cs769_02`, repo `.../Multi-Hop-Project`, runs `.../mhtable-runs`. Windows `scp` needs absolute remote paths (no `~`). Move results with tar (`--exclude='*.pt'`) + scp into `runs/prajna/`; push checkpoints with `python -m mhtable.storage push runs <dir> prajna/<run>`.
  - **Shaping caveat (2026-11-23):** with Φ(terminal) = 0 and γ = 1, R1 is invisible to outcome GRPO (R1 ≡ R0 exactly) and reaches PPO only through λ = 0.95 truncation and the critic. Process GRPO (`algo.advantage: process`, λ < 1) is the ablation that sees the shaping. 127 tests pass.
  - **P9 results (2026-11-23, PLAN P9 table):**
    - All runs are on HF. Test EM vs BC 28.0 / 26.8: every RL arm lands at 27.7–28.8 / 26.8–28.7, i.e. ≤ +1.9 pts.
    - Only relaxed PPO-R1/R0 and GRPO-R0 beat BC significantly (paired bootstrap). Arm-vs-arm differences are noise; GRPO R1 ≈ R0; process GRPO gives no gain.
    - The policies stay 74–89% identical to BC.
  - **Big-step + from-scratch runs done 2026-11-23 (PLAN P9 table; `grpo-r1-relaxed-scratch-s0` not on HF):** bigger steps move policies 0.4–1.8 KL from BC with no EM gain; from scratch reaches best dev 27–30 but test 4–7 pts below BC; shaping R1 beats R0 from scratch (+2.5 / +2.9, significant); PPO ≈ GRPO everywhere.
  - **Next:** slides (flow agreed 2026-11-23, 9 slides + appendix), plots + report text; the molab inference demo (ideated in PLAN P9, not coded); optionally a larger-step RL run.
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
  smoke test, bge-m3 embedding cache, P4 gate), `reader_ablation.py` (7B/14B x prompt p1/p2 on
  gold sub-tables, with throughput), `track_c.py` (BC with bge-m3 + baselines + BC gate).
- Laptop: Windows, RTX 4060 8 GB, slow network (~370 KB/s) — development and unit tests only;
  never download BIRD databases (32 GB) locally.

## Prajna HPC (main compute once login works; molab until then)

Source: `Updated User Manual for Prajna (AiML) HPC.pdf` in the project root. Re-read it when in
doubt; this section summarises it and our rules on top of it.

**What it is.** IIT Bombay's Prajna (AI-ML) cluster: Rocky Linux 9, x86_64, SLURM 23.11,
Lustre parallel filesystem (2 PB), InfiniBand. GPU partitions:

**Measured on first login (2026-11-21, `sinfo` / `scontrol` / `sacctmgr`):**

| Partition | Nodes | GPUs/node | CPUs/node, RAM | Max wall | QoS (same name) — max jobs per user |
|---|---|---|---|---|---|
| `dgx` | 9 (cn11–19) | 8× A100 | 256, ~2 TB | 6 days | `dgx` — 4 |
| `dgx-mpi` | 3 (cn11–13, shared with dgx) | 8× A100 | 256, ~2 TB | 6 days | `dgx-mpi` — 1 |
| `a40` | 19 (cn21–39) | 4× A40 48 GB | 64, ~500 GB | 4 days | `a40` — 3 |
| `l40` (default) | 6 (cn40–45) | 8× L40S 48 GB | 32+, ~450 GB | 2 days | `l40` — 4 |
| `interactive` | 3 (cn11-dgx, cn22-a40, cn41-l40) | 8 or 4 | 64+, ~500 GB | 4 h | `interactive` — 2 |
| `debug` | 1 (cn39-a40) | 4× A40 | 64, ~500 GB | 30 min | `debug` |

QoS `l4`, `gh200` and `normal` are also granted, but no `l4`/`gh200` partition is visible in
`sinfo` (the manual's L4 nodes are not schedulable for us). Each partition only allows its own
QoS (`AllowQos=<name>`), so every job sets **`--partition=X --qos=X`** with the same name.
Accounts: `cs769_02` (ours; also `cs769`), **MaxJobs 12** in total across partitions.

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

**Confirmed 2026-11-21:**
- Partitions / QoS / limits: table above. Interactive GPU sessions **are** available
  (`srun -p interactive --qos=interactive --gres=gpu:1 -t 01:00:00 --pty bash`, ≤ 4 h, ≤ 2 jobs).
  **`srun` works only on `interactive`** ("Interactive jobs are only allowed on partition
  'interactive'"); every other partition, `debug` included, takes `sbatch` scripts only.
- QoS limits (`sacctmgr show qos`, 2026-11-21) — per user: submitted / running; whole QoS:
  submitted / running. `a40`: 6 / 3, QoS 100 / 80. `l40`: 5 / 4, QoS 70 / 56. `interactive`:
  2 / 2, QoS **5 / 3 for the whole cluster** (hard to get). `debug` shows no limits, yet
  `sbatch` to it failed with `AssocGrpSubmitJobsLimit` — treat `debug` as unusable for us.
  So job arrays must use `%3` (a40) or `%4` (l40) and stay within 6 / 5 submitted tasks.
- `/home`: `lfs quota` shows **no user quota** (limit 0 = unlimited; 224 KB used).
- SLURM account: `--account=cs769_02`.
- Filesystems: `/home` is the 1.3 PB Lustre (73% full; `quota -s` reports nothing, so no user
  quota was shown — check `lfs quota -h -u $USER /home`). **`/scratch` is not Lustre** here: a
  250 GB filesystem (`slurm:/scratch`, 212 GB free) — `lfs quota` fails on it. Our data repo is
  ~33 GB (BIRD DBs 32 GB) + model weights ~15–30 GB + BIRD embeddings 5.4 GB: fits, but check
  whether `/scratch` is shared by all users or node-local before relying on it; the safer
  default for caches is a folder under `/home`.

**First batch job (2026-11-21, `sbatch` on `a40`, node cn25-a40) — works with
`--partition=a40 --qos=a40 --account=cs769_02`:**
- **GPU driver 570.86.15, CUDA 12.8** (A40 = Ampere, bf16 supported). molab ran vLLM 0.34 with
  torch built for CUDA 13; on Prajna torch/vLLM must be **CUDA 12.x builds** — a cu13 wheel will
  not run on this driver. vLLM's default PyPI wheel is cu13: it imports on the login node but fails
  on a GPU node with `libcudart.so.13: cannot open shared object file` (reader_smoke 2026-11-22,
  with vllm 0.24.0 + torch 2.11.0+cu128). vLLM 0.24.0 publishes only cu129 and cu130 builds, so
  `setup_reader.sh` installs `vllm==0.24.0+cu129` from `https://wheels.vllm.ai/0.24.0/cu129` with
  torch cu129 (a 12.9 runtime on the 12.8 driver relies on CUDA minor-version compatibility).
- `/scratch` on the compute node is the same `slurm:/scratch` (250 GB, same usage as on the login
  node), so it is shared between login and compute nodes — but small; prefer `/home` (no quota).
- **Internet on compute nodes: none (confirmed 2026-11-22**, reader_smoke on cn41-l40: curl to
  huggingface.co gave `000` / `FAILED`). Install packages and download data + models on the login
  node into `/home`; jobs run with `HF_HUB_OFFLINE=1`.

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
