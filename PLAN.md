# Project Plan (v2, 2026-10-09)

The two source-of-truth documents are this file and the report PDF
(`Multi-Hop_Table_RL_Project_Report.pdf`), plus `DESIGN.md` for the architecture (modules,
interfaces, storage). The report defines the problem, MDP, rewards, metrics and algorithm arms
(Parts A–E); this file defines data, build order and the engineering rules, and overrides the
report where they differ (§0). The older `PROJECT.md`
and `TASK_01_DATASET.md` were removed on 2026-10-09; their still-valid rules are in §6.

## 0. What changed and why

| Decision | Choice | Reason |
|---|---|---|
| Training data | Real questions with gold SQL: Spider, BIRD-train, plus MMQA's own train split. **No LLM-generated corpus.** | About 6–8k usable questions already exist (`src/analysis/action_fit.py`). Generation costs GPU time and loses 30–50% of its output to verification. |
| Split | **One database-level split shared by Spider and MMQA** (60/10/30) | MMQA was built from 139 of 140 Spider-train DBs and all 20 Spider-dev DBs, so splitting by dataset would leak test schemas. |
| Storage | **All datasets on private HuggingFace repos**; local disk is only a cache (`DESIGN.md` §4) | Uniform access from laptop and molab; versioned, pinned revisions. |
| BIRD | Sourced, built and trained on **molab** | BIRD is 33 GB; the laptop's link is about 370 KB/s. The laptop never fetches BIRD DBs. |
| Action set | Relaxed tier, with a `--tier strict` switch | Strict is a subset; every instance carries `strict_ok` / `relaxed_ok` flags. |
| Reader | Behind an interface: dummy, `sufficiency` mock, HF-transformers, vLLM | Smoke tests locally on CPU / 8 GB; main runs on molab. Reader comparisons later. |
| Policy net | MLP towers first, transformer encoder pluggable | Simpler to debug the first PPO/GRPO runs. |
| Algorithms | Pluggable advantage estimators and update rules | First milestone is BC + PPO + GRPO with R0/R1. The full report grid is added without touching rollout/env code. |
| Motivation | Reframe around join selection and long-tail tables (G-2 failed) | Median MMQA table has ~15 rows; 84–92% of questions fit in 4,096 tokens. BIRD and distractor tables supply the hard cases. |

## 1. Where things run

| Where | What |
|---|---|
| Laptop (CPU, RTX 4060 8 GB) | All code; unit tests; Spider + MMQA builds (fetched from HF); env/oracle/BC; PPO/GRPO smoke runs with the mock reader or a ≤1.5B reader |
| **Prajna HPC (main compute from P4)** | SLURM batch jobs on A100 / A40 / L40S / L4 GPUs; reader serving, training runs, seed/arm grids as job arrays, BIRD rebuilds. Rules and open questions: `CLAUDE.md` § Prajna HPC; source: `Updated User Manual for Prajna (AiML) HPC.pdf` |
| molab (main compute until Prajna access works; GPU, 12 h sessions, 90 min idle shutdown) | Data sourcing to HF; BIRD builds; embedding cache; reader-in-the-loop PPO/GRPO; evaluation |
| GitHub `advay333/Multi-Hop-Project` (public) | Code. Notebooks clone it. |
| HuggingFace (private) | `mhtable-raw` (sources), `mhtable-data` (built data + SQLite DBs), `mhtable-cache` (embeddings, reader answers), `mhtable-runs` (checkpoints, results) |

**Rule:** notebooks contain no logic. Each one is a marimo `.py` that clones the repo, installs
it, pulls data from HF, runs one `python -m mhtable.<entry> --config ...`, and pushes outputs
back to HF. They checkpoint often and can resume, because sessions die.

## 2. Code layout (target)

Full module contracts are in `DESIGN.md` §6; this is the file map.

```
src/mhtable/
  config.py         dataclass configs loaded from YAML; every switch lives here
  registry.py       name -> class registries: readers, encoders, policies, algorithms, selectors
  storage.py        fetch / push / db_path against the HF repos (local = cache only)
  types.py          Instance, GoldRecord, Action, State, Step, Trajectory
  data/
    norm.py         norm_value() — the ONE normaliser (§6 rule 3)
    schema.py       SchemaGraph (tables, PK/FK, inferred edges)
    loaders/        spider / bird / mmqa -> (Instance, gold_sql)
    mmqa_db.py      materialise per-instance MMQA tables to SQLite (cached by schema signature)
    split.py        shared database-level split
    build.py        one command: mhtable-raw -> mhtable-data (instances, gold, splits, DBs, indexes)
  oracle.py         sqlglot -> gold edges/anchors, prefix execution -> E_t*, canonical trajectory
  env/
    actions.py      Action types, candidate enumeration (strict | relaxed), masking
    edges.py        JAR-style edge inference (omega) vs gold-FK edges (flag)
    cells.py        anchor cell index (distinct values per column, ID columns excluded)
    env.py          TableEnv: reset / candidate_actions / step / materialise(budget)
  readers/          base.py (interface + cache) | mock.py | hf.py | vllm.py
  encoders/         base.py | st.py (bge-m3 etc.) | cache.py (npz/HF lookup)
  featurise.py      state/action -> tensors; firewall assertions; header-ablation flag
  policies/         base.py | mlp.py | transformer.py (later)
  firewall.py       assert_gold_free(): runtime check used by featurise and the reader prompt
  sufficiency.py    gold-aware: path_match, sufficient (mock reader's rule; diagnostic)
  rewards.py        R0 sparse, R1 Jaccard potential, token cost
  dataset.py        gold-aware: (Instance, GoldRecord) pairs for a split and tier
  rollout.py        THE shared collector (sample|argmax, oracle_mode, selector)
  algos/            base.py (advantage + update interface) | bc.py | ppo.py | grpo.py
                    later: ppo_kl.py | reinforce.py | vineppo.py
  selectors/        baselines: truncate | random_rows | heuristic | oracle | policy
  metrics.py        EM_norm, set-F1, join-path acc, sub-table coverage, edge P/R, tokens
  train.py          entry point for BC / RL training
  evaluate.py       entry point for every selector
configs/            env.yaml, reader/*.yaml, algo/*.yaml, experiments/*.yaml
notebooks/          source_data.py, build_data.py, embed.py, train.py, evaluate.py  (marimo, molab)
tests/              pytest; runs on CPU in < 2 min
```

## 3. Phases

Each phase ends with a **gate**: a concrete check that must pass before building on top.

### P0 — Repo skeleton (laptop) — DONE 2026-10-11 (gate passed on molab)
- `pyproject.toml` (installable `mhtable` package), config loader, registries, `storage.py`,
  `types.py`, pytest.
- Create the four private HF repos; `notebooks/source_data.py` runs
  `python -m mhtable.data.source` on molab, which downloads Spider, the MMQA mirror and BIRD,
  verifies checksums (`SOURCES` in `src/mhtable/data/source.py`), and pushes to `mhtable-raw`.
- `src/analysis/action_fit.py` moved to `mhtable.analysis.action_fit`.
- **Gate:** `pip install -e .` and `pytest` pass on the laptop and in a molab notebook;
  `storage.fetch` retrieves a raw file on both.

### P1 — Data layer (molab builds; laptop dev runs; all output to HF) — DONE 2026-10-12 (`data-v1`)
- `norm_value`, `SchemaGraph`, `sqlshape` (tier analysis), loaders for Spider / MMQA / BIRD
  producing a public `Instance` and a separate `GoldRecord` (`DESIGN.md` §5).
- MMQA: 659 SQLite files (one per schema; contents verified identical), mojibake repaired,
  keys borrowed from the matching Spider DB (649 of 659), MMQA's own key lists otherwise.
- Shared 60/10/30 split by DB over Spider ∪ MMQA, balanced over 2-/3-table MMQA questions;
  Spider-only DBs → train; BIRD-train DBs → train; BIRD-dev → `bird_dev`.
- `notebooks/build_data.py` runs `python -m mhtable.data.build --family {spider_mmqa,bird} --push`.
- Local full Spider+MMQA build (not pushed): 2.5 min; relaxed-tier usable questions —
  train 2,179 Spider + 1,266 MMQA; dev 225 + 234; test 711 + 667 (`data/DATASET_CARD.md`).
- BIRD build: 10,962 questions over 80 DBs; relaxed-tier usable — train 5,179, bird_dev 870.
- **Gate (passed):** both families pushed from molab and tagged `data-v1` (964 files, 32 GB);
  molab Spider+MMQA stats identical to the local build; splits instance-disjoint; 20 random gold
  queries re-execute to their stored answers from the uploaded DBs.

### P2 — Oracle (runs inside the data build) — DONE 2026-10-15 for Spider+MMQA (`data-v2`); BIRD fixes in `data-v3` (tagged 2026-11-17)
- `oracle.py`: gold structure (tables, edges, anchors with literal values, monotone flag),
  canonical BC trajectory (selectivity-aware start, BFS joins, anchors after their table's
  JOIN), per-step entity sets (row count + capped rowids), Jaccard potential Φ.
- `sqlrender.py`: state -> SQL, shared with the P3 environment.
- Checks: reconstruction (replayed plan == gold FROM/JOIN/WHERE rows), join uniqueness,
  timeouts/errors. Both tiers tracked: `stats/<family>.json` -> `oracle.{strict,relaxed}`.
- Local Spider+MMQA result: pass rate strict 99.7% / 99.4%, relaxed 98.5% / 97.8%; zero
  reconstruction mismatches; all failures are `join_not_key`.
- **Gate:** oracle passes ≥95% of eligible questions in each tier — passed for Spider and MMQA
  (`data-v2`, identical to the local run; 20 random trajectories replay to their stored counts).
  **BIRD: 93.7% strict / 91.1% relaxed — accepted as-is for now, see §4a.**

### P3 — Environment (laptop) — DONE 2026-10-16 for Spider+MMQA (index pushed); BIRD G-6 (data-v3): strict 87.4% passes, relaxed 79.5% accepted
- `env/text.py`, `env/index.py`, `env/actions.py`, `env/env.py` (`DESIGN.md` §6.4);
  `analysis/coverage.py` for G-6; `notebooks/build_index.py` builds and pushes `index/` on molab.
- Inferred joins (JAR ω) by default, `edges: declared` as ablation; threshold ω ≥ 0.6667 keeps
  98% of gold joins on train. Lexical similarity until the P4 encoder exists.
- New oracle check `gold_relation_empty` (139 strict / 203 relaxed train questions on Spider's
  empty `scholar`/`yelp`/`academic`/... databases); applied at load time until the next rebuild.
- **Gates (Spider+MMQA train, data-v2):** G-6 reachability strict **89.6%** (2,364/2,638),
  relaxed **85.1%** (2,679/3,149) — passed; dev: 87.8% / 82.5%. Random policy: 1,000 episodes
  per tier, 0 crashes, |A(s)| mean 7–8, max 21; episode lengths 2–5.
- Remaining misses are mostly semantic (e.g. "male" -> 'M', "European" -> continent 2) — the
  P4 embedding scorer may recover some; they are reported, not forced.
- BIRD: index pushed; G-6 below gate (train 80.5% strict / 65.9% relaxed) — pending, see §4a.

### P4 — Readers, encoders, metrics (laptop + molab)
- Reader interface with on-disk cache keyed by hash(reader, prompt). Implementations:
  - `mock`: returns the gold answer iff the sub-table is sufficient (a free reward proxy for debugging RL);
  - `hf`: transformers, any size, for local smoke tests;
  - `vllm`: for molab.
- Encoder interface + embedding cache (questions, `table: cols`, column names, sampled
  distinct cells). Built once on molab, pushed to HF; `featurise` is then a dict lookup.
- Metric suite (report Part D).
- First molab notebook confirms GPU type, vLLM install, and reader throughput (calls/hour).
- **Gate:** the oracle selector + real reader runs on 200 dev questions end-to-end on molab.

### P5 — Shared rollout, rewards, policy, algorithm interface (laptop) — DONE 2026-11-17 (built before P4; gate passed)
- **Built:** `featurise.py`, `policies/{base,mlp}.py`, `rewards.py`, `rollout.py`, `algos/base.py`,
  `firewall.py`, `sufficiency.py`, and the thin P4 slices P5 needs: `readers/{base,prompt,mock}.py`
  (interface, shared prompt, cache + call counter, mock reader), `encoders/{base,cache}.py`
  (interface, `hash` test encoder, `EmbCache`), `metrics.py` (EM_norm, set-F1 only).
  **Still P4:** `hf` / `vllm` readers, the bge-m3 encoder + `build_cache`, the full metric suite,
  the reader's tokenizer in `materialise`.
- **Gate passed:** `tests/test_learning.py` (13 tests; 93 in total) — shaping telescopes, GAE at
  γ = 1, masking + permutation invariance + near-uniform start, firewall (runtime check and
  import check), plus the worked example of report Figure 1, curriculum start, mock
  reader/sufficiency, reader cache, and a miniature BC fit that recovers the gold path.
- **Real-data smoke (laptop, ad-hoc script outside the repo, cached data-v2 Spider+MMQA, 300
  random relaxed train questions, mock reader, hash encoder):** following the gold trajectory
  gives path match 87.0% (261/300; consistent with G-6 reachability 86.3%) and sufficient 81.7%
  (245/300): 20 gold-path sub-tables are cut by the 4,096-token budget and so count as
  insufficient, 4 off-path ones are still sufficient. An untrained policy gives 7.7%.
  Telescoping error ≤ 1e-16; MLP policy 2.30M parameters; about 0.1–0.3 s per episode (look-ahead
  COUNTs dominate; group rollouts recompute them per member — to optimise for GRPO in P9).
  A frozen cache built from `featurise.required_texts` served all 300 episodes with no miss;
  it held 213,695 distinct texts for 168 databases (mostly cell values) — the size of the P4
  embedding job. The mock reader's answer text scores EM 1 on all 10,988 usable gold answers.
- Original task list:
- `rollout.collect(env, policy, reader, instances, sample_or_argmax, oracle_mode, selector)`.
- `rewards.py`: R0 (β = 0), R1 (Jaccard potential, Φ(terminal) = 0), cost λ_c tok/B.
- `policies/mlp.py`: state tower + action tower, scaled dot-product logits, learnable
  temperature, separate scalar normalisation, small final-layer init (§6 rule 7).
  Shares an interface with the later transformer.
- `algos/base.py`: `compute_advantages(batch)` + `update(batch)`. Every RL arm implements only these two.
- Firewall assertions (gold fields can never reach featurise or the reader prompt).
- **Gate:** unit tests for telescoping shaping (sum = 0), GAE at γ = 1, masking, firewall.

### P6 — Baselines + oracle ceiling (laptop with mock, then molab with the real reader)
- Selectors: B1 truncate, B2 random rows, heuristic (greedy cosine + top-k), oracle sub-table.
- **Gate:** numbers on dev for each; the oracle ceiling tells us how much headroom the reader leaves.

### P7 — Behaviour cloning (laptop CPU) — CODE DONE 2026-11-17 (built before P6; gate still open)
- **Built:** `dataset.py`, `algos/bc.py` (`demonstrations`, `BC`), `train.py` (entry point,
  checkpoints, dev evaluation by rollout), `configs/experiments/bc.yaml`; 95 tests pass.
  Run now (laptop, offline, against the HF-cache snapshots of data-v2 and its index):
  `HF_HUB_OFFLINE=1 python -m mhtable.train --config configs/experiments/bc.yaml --data-dir
  <snapshot with instances/gold/schemas/splits/db> --index-dir <snapshot with index/>
  --run-id bc-strict-s0 env.tier=strict` (same with `relaxed`). The bare command without
  `--data-dir` reads HuggingFace at `data.revision` — use it only once `data-v3` is tagged and
  pinned in the config. Each run's `config.yaml` records the data/index folders, the commit and
  whether the checkout was dirty (the two runs below predate that and ran on uncommitted code).
- **First runs** (laptop CPU, cached data-v2 Spider+MMQA, seed 0, **`hash` lexical encoder, mock
  reader** — a pipeline check, not a result to report; ~12 min each, best of 30 epochs by dev
  join-path accuracy = epoch 20 in both):

  | Tier | Train demos (reachable / questions) | Dev join-path acc. (untrained → BC) | Dev sufficient | Train-subset path acc. (300) | Dev teacher-forced step acc. |
  |---|---|---|---|---|---|
  | strict | 2,402 / 2,638 | 0.5% → **31.5%** (n = 394) | 42.9% | 77.0% | 60.9% |
  | relaxed | 2,719 / 3,149 | 0.4% → **25.3%** (n = 458) | 37.3% | 72.3% | 57.8% |

  Dev slices (strict / relaxed join-path): Spider 40.7% / 38.2%, MMQA 20.6% / 12.9%; 2-table
  36.1% / 30.9%, 3-table 14.3% / 4.2%. Dev numbers count questions whose canonical gold path is
  not offered as failures (46 strict / 78 relaxed; no path match occurred on any of them), so
  canonical-order reachability (88% / 83%) is roughly the ceiling.
- **Order matters for reading the numbers:** join-path accuracy is order-free, teacher-forced
  accuracy demands the oracle's canonical order. Of the greedy dev path matches, 68 of 124
  (strict) and 65 of 116 (relaxed) follow the canonical order; the other 56 / 51 reach the same
  joins and filters in a different order. So step accuracy understates BC, and BC's labels are
  ambiguous between equivalent orders (a known BC weakness, report A.8 — RL is free of it).
- **Reading:** the pipeline learns (0.5% → 25–32%), but the train–dev gap is large (72–77% vs
  25–32%), and teacher-forced dev loss rises from epoch 5 on in both tiers while train loss
  falls — overfitting. Hypothesis (untested): dev databases are unseen and the `hash` encoder
  only sees shared words, so the policy memorises training schemas; bge-m3 embeddings (P4)
  should narrow the gap. If they do not, add regularisation (dropout, label smoothing, smaller
  d) and BIRD demonstrations. 3-table questions are the weak slice.
- **Open:** the gate needs the P6 heuristic selector to compare against; rerun BC with the real
  encoder once the P4 embedding cache exists. Memory note: demonstrations hold full feature
  arrays — fine at 64-d (≈ 0.2 GB), but with 1024-d embeddings ≈ 1–3 GB for Spider+MMQA(+BIRD):
  store float16 or featurise per minibatch before the bge-m3 runs.
- Original task list:
- Cross-entropy on canonical oracle trajectories; this is also the RL initialisation.
- **Gate:** BC beats the heuristic on dev join-path accuracy (mock reader first, then the real reader).

### P8 — PPO (laptop with mock → molab with the real reader) — CODE DONE 2026-11-17; gate G-8 NOT YET READ
- **Built:** `algos/ppo.py`, the RL loop `train.run_rl` (curriculum, `--resume`, `train.jsonl` +
  `metrics.jsonl`), `selectors/oracle.py` (gold-path reference), candidate memoisation in
  `rollout.collect` (`cand_cache`), `configs/experiments/ppo.yaml`. 98 tests pass, including PPO
  overfitting the fixture question through the real entry point (G-8 in miniature).
- **Runs started, results not yet read or recorded** (laptop, data-v2 Spider+MMQA relaxed, `hash`
  encoder, mock reader): `runs/g8-r1`, `runs/g8-r0` (G-8: 20 reachable questions, random start,
  150 iterations; logs `runs/g8-R1.log`, `runs/g8-R0.log`) and `runs/ppo-relaxed-r1-s0`,
  `runs/ppo-relaxed-r0-s0` (from `runs/bc-relaxed-s0/policy.pt`, 100 iterations; logs
  `runs/ppo-R1.log`, `runs/ppo-R0.log`). Read each run's `result.json`: G-8 passes if
  `train_pool_greedy.return` ≈ `train_pool_gold_path.return`.
- Original task list:
- PPO-Clip with GAE, value clipping, entropy, KL to BC (report Table 4 defaults).
- **Gate G-8:** overfits 20 questions to return ≈ 1.0 (mock reader). Then R0 vs R1 runs.

### P9 — GRPO + the first result
- GRPO (G = 8 rollouts per question, group-normalised return, KL to BC); log the zero-advantage fraction.
- 2×2 grid {PPO, GRPO} × {R0, R1}, curves plotted against **reader calls**, not gradient steps.
- **Gate:** the 2×2 table on dev; then test numbers with 3 seeds once the setup is frozen.

### P10 — Extensions, in priority order (pick as time allows)
1. More arms: REINFORCE, PPO-KL, VinePPO (each is one file in `algos/`).
2. Remaining baselines: prompted planner (B5), SQL-extract-then-read (B6).
3. Robustness: header ablation, 2- vs 3-table, distractor sweep, BIRD-dev transfer.
4. Transformer policy; reader comparison; budget sweep.

## 4a. data-v3 (tagged 2026-11-17) and what is still open

**`data-v3` is tagged** on mhtable-data (data + index together), built on molab with
`notebooks/rebuild_v3.py`. It contains the 4 BIRD oracle fixes (composite-key joins, outer joins
rejected, `WITHOUT ROWID` entity sets, separate oracle timeout), `gold_relation_empty` as an
eligibility filter, `col LIKE 'text'` -> `=`, and the revised action set / index (DESIGN §6.4).

- **Pools (oracle-verified questions, strict / relaxed):** Spider train 1,707 / 1,944, MMQA train
  942 / 1,219, BIRD train 3,038 / 4,679; dev 394 / 458; test 1,172 / 1,361 (Spider 642 / 701,
  MMQA 530 / 660); bird_dev 643 / 832. All match the projections.
- **G-6 reachability (`analysis.reach`, data-v3 main):** Spider+MMQA train 91.0% / 86.2% (pass),
  dev 88.3% / 83.0%; BIRD train 87.4% (pass) / **79.5%** (below 85%), bird_dev 80.1% / 73.6%.
- **User decisions 2026-11-17:** (1) accept the BIRD oracle pass rate (~93% relaxed, gate 95%; the
  gap is genuine fan-out joins in BIRD's gold SQL; failures are dropped, never repaired);
  (2) accept BIRD relaxed reachability 79.5% (gate 85%). Spider+MMQA passes every gate.
- **Not verified:** the 4a pass-rate table (per-tier `fail_reasons`) was lost when the molab
  session died, so `oracle_no_rowid` = 0 was not seen directly; the eligible counts match the
  projection. To re-check: `stats/bird.json` on HF (if the build pushed `stats/`).
- **Open: BIRD real-SQL check (step 4c) not completed.** `analysis.coverage --family bird --limit
  500` ran 90+ min without finishing tier 1: ~50 min inside `movie_platform` alone, then stalled on
  `talkingdata` (both have tables of millions of rows). Look-ahead COUNTs hit the 2 s timeout
  (`EnvConfig.lookahead_timeout_s`) at up to 25 candidates per step. Crash-freedom of composite
  JOINs / WITHOUT ROWID / evidence anchors on real BIRD DBs is therefore **unverified**, and **RL on
  BIRD will be equally slow** until fixed. Candidate fixes: SQLite indexes on join/anchor columns
  (changes the DB files -> `data-v4`), a much shorter look-ahead timeout or row-count cap for huge
  tables, or excluding the largest DBs. Spider+MMQA are unaffected.
- **BIRD relaxed coverage**, if it must reach 85% later: remaining blockers are values outside the
  10k cell index, non-inferred joins, join/numeric ranking, rare LIKE shapes; candidates are
  embedding similarity (P4) and a runtime exact-value probe.

## 4. Things we will measure early because they can change the plan

- **G-6 coverage** (P3): if < 85%, widen the action set before training.
- **Oracle ceiling** (P6): if the reader barely beats truncation even on gold sub-tables, the reader is the bottleneck. Change the reader before investing in RL.
- **Reader throughput on molab** (P4): sets how many PPO iterations a 12 h session allows.
- **BIRD token-budget fit** (P3, needs the BIRD build): decides how much BIRD actually contributes.

## 5. Success criteria

Stated against our own matched-reader baselines, never against published frontier-model
numbers (a frozen 7B reader sits far below them in absolute terms).
- **Minimum:** environment + oracle + baselines working; shaped PPO ≥ BC; a clean answer to
  PPO vs GRPO under a matched reader-call budget.
- **Target:** beat the strongest non-learned baseline at ≤ 50% of reader tokens; higher
  join-path accuracy; gains hold on 3-table questions and on BIRD.
- **Report headroom closed**, not raw EM: (policy − BC) / (oracle ceiling − BC).

## 6. Engineering rules (carried over from the removed specs)

1. **No database-specific integer IDs** anywhere (cache keys, features, JSON). Everything is
   referenced by `table.column` strings and featurised from names and content, so a policy
   transfers to unseen databases.
2. **Permutation invariance** over sets (frontier values, candidate actions): pool, never
   concatenate in order.
3. **One normaliser** `norm_value()` used everywhere: strip, casefold, collapse whitespace;
   `"7" == "7.0" == " 7 "`; NULL → `"<null>"`; dates → ISO-8601. A mismatch silently zeroes the
   shaping reward.
4. **Firewall:** gold SQL, gold answer, gold entity sets, join path and predicates are
   training-time reward signals only. Assert in `featurise` and the reader prompt builder that
   none are present; `oracle_mode` ∈ {`reward` (train only), `metric_only` (eval), `off`}.
5. **One rollout implementation** shared by training and every evaluation mode; behaviour
   differs only through flags.
6. **Same materialiser, reader, prompt and budget for every selector** — only the selector
   changes between compared rows.
7. **Network details:** normalise scalar features separately from embeddings and `log1p` all
   count-like features; scale dot-product logits by 1/√d with a learnable temperature;
   initialise the last layer small so the policy starts near uniform.
8. **Reader calls are batched and cached** by hash of (reader, prompt) at temperature 0;
   RL curves are plotted against reader calls, not gradient steps.
9. **Coverage is measured and reported, not forced:** aggregation is the reader's job (gold
   entity sets come from FROM/JOIN/WHERE only); unrepresentable instances are dropped and
   counted with a reason.
10. **Schema-vocabulary leakage:** keep a header-ablation flag in the featuriser (column names
    → `col_0..col_n`) for the robustness study.
11. **Seed everything; regeneration is one command.**

## 7. What you need to set up

- A HuggingFace account token with write access to private repos (stored as a molab secret).
- A molab account with GPU access enabled.
