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
| molab (GPU, 12 h sessions, 90 min idle shutdown) | Data sourcing to HF; BIRD builds; embedding cache; reader-in-the-loop PPO/GRPO; evaluation |
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
  rewards.py        R0 sparse, R1 Jaccard potential, token cost
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

### P2 — Oracle (runs inside the data build) — DONE 2026-10-15 for Spider+MMQA (`data-v2`); BIRD gate waived, pending
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

### P3 — Environment (laptop) — code done 2026-10-15; index push pending
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
- BIRD: index + coverage pending (needs the BIRD DBs, molab).

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

### P5 — Shared rollout, rewards, policy, algorithm interface (laptop)
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

### P7 — Behaviour cloning (laptop CPU)
- Cross-entropy on canonical oracle trajectories; this is also the RL initialisation.
- **Gate:** BC beats the heuristic on dev join-path accuracy (mock reader first, then the real reader).

### P8 — PPO (laptop with mock → molab with the real reader)
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

## 4a. Pending before final training runs

- **BIRD oracle gate (P2).** `data-v2` BIRD passes 91.1% relaxed / 93.7% strict (gate 95%).
  Usable as-is: 4,654 relaxed train + 833 bird_dev questions with verified oracle labels.
  Known causes and fixes, to apply and rebuild BIRD as `data-v3` before the final runs:
  1. `join_not_key` (463): the uniqueness check tests composite-key joins column by column
     (e.g. `tmID AND year`); test the column combination instead. The rest are genuine
     many-to-many joins (e.g. `superstore`) and stay rejected.
  2. LEFT JOINs (27, caught by the reconstruction check): reject structurally in `sqlshape`.
  3. `WITHOUT ROWID` tables (33, all `language_corpus`): key entity sets on the primary key.
  4. Timeouts (15): longer timeout for the oracle stage.
- **Next data rebuild (`data-v3`)** also bakes in the P3 oracle check `gold_relation_empty`
  (loaders apply it until then), and the BIRD index + G-6 coverage need a molab run.

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
