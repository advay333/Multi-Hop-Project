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
| Kaggle (free; added 2026-11-18) | Second GPU box alongside molab: 2× T4 16 GB, 4 CPU cores, 29 GB RAM, 12 h per session, 20 GB saved `/kaggle/working`, 20 min idle timeout for interactive sessions but "Save & Run All" runs in the background; weekly GPU quota set per account (commonly ~30 h, check the account page). Roles: embedding cache, P6 baselines / evaluation, part of the P9 grid. Not for BIRD builds (disk) or CPU-bound RL debugging (4 cores) |
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

### P4 — Readers, encoders, metrics (laptop + molab) — CODE DONE 2026-11-18; gate needs the molab run
- **Built (107 tests pass + 2 `network` tests with tiny real models):** `readers/llm.py` (shared
  chat prompt, row-cut to the context window, answer clean-up, `reader_id` = backend + model +
  precision + greedy + max tokens + prompt version), `readers/vllm.py`, `readers/hf.py`,
  `CachedReader.seconds`; `env/tokens.py` + `env.tokenizer` (null | model | `reader`) — the
  budget is counted with the reader's tokenizer in `materialise` (look-ahead stays
  character-based); `encoders/st.py` (bge-m3), `encoders/build_cache.py` (resumable parts,
  float16, manifest, `--push` to `mhtable-cache embeddings/<encoder>/<family>.npz`), `EmbCache`
  as one float16 matrix, `encoder.cache: hf:...` in `train.build_featuriser`; the metric suite
  (`metrics.py`: answer_in_subtable, join-edge / filter P/R, `summarise` by source / 2-3 tables /
  category / monotone, `paired_bootstrap`; `sufficiency.entity_overlap` per hop; rollout
  `detail=True`); `evaluate.py` (oracle | policy selector, both tiers, predictions + result,
  reader calls/hour, `--reported` refuses mock/oracle); `configs/experiments/eval.yaml`
  (Qwen2.5-7B-Instruct, vLLM, float16, max_model_len 6144); `python -m mhtable.storage push`;
  molab notebook `notebooks/p4_reader.py` (setup → 5-question smoke → bge-m3 cache → gate on
  N dev questions, both tiers → push).
- **Laptop dry run** (data-v2 Spider+MMQA dev, oracle selector, mock reader, 60 questions per
  tier): EM 93.3% strict / 83.3% relaxed, but **answer_in_subtable only 69% / 48%**. Even on the
  gold path, the answer's columns are often not shown: of the 40 strict Select/List questions
  on the gold path, 15 sub-tables miss them (checked in the sub-tables: "names of
  entrepreneurs, companies, money requested" shows no `people.name`; "store with the largest
  area" no `store.area_size`; "ids, names, FDA status of medicines" no `medicine.name`) — the
  projection shows join keys, filter columns and the top-4 *lexically* similar columns only.
  The mock reader hides this; a real reader cannot answer without the column. **Fixed
  2026-11-18 (user decision):** `EnvConfig.project_all_columns` (default on) shows every other
  column of the visited tables, most question-relevant first, as long as all rows still fit the
  budget (estimated), never fewer than the old top 4. Same dry run after the fix:
  answer_in_subtable **90.8% strict / 87.5% relaxed**, and 0 of the 40 / 42 gold-path
  Select/List questions miss their answer column (what remains is Count/Numerical answers,
  which the reader computes); mean tokens 180 → 298 / 305 → 447. The projection only affects
  `materialise`, not the candidate actions, so G-6 reachability and BC are unchanged; the
  mock-reader PPO runs of P8 predate the fix (wider rows → more tokens, more overflow). The P4 gate run reports `answer_in_subtable` and
  `em_when_answer_shown` to separate projection loss from reader errors.
- **Gate run (molab, 2026-11-19; `mhtable-runs/runs/p4-gate-oracle-dev200`; code 8d359ee, data-v3,
  Qwen2.5-7B-Instruct bf16 via vLLM 0.34, RTX PRO 6000):** end-to-end passes. Oracle selector,
  200 dev questions per tier:

  | Tier | EM | set-F1 | path match | EM on path | answer shown | EM when shown | tokens |
  |---|---|---|---|---|---|---|---|
  | strict | 52.0% | 70.4% | 89.5% | 55.9% | 90.9% | 54.9% (n = 175) | 485 |
  | relaxed | 49.0% | 66.6% | 82.0% | 55.5% | 86.2% | 51.2% (n = 168) | 558 |

  By category (strict EM): Select 69%, List 37%, Count 60%, Numerical 42%; monotone 64% vs
  non-monotone 41%. **The reader is the main loss:** with the gold path and the answer visible,
  it is right only ~55% of the time. Of the 79 strict misses with the answer shown: 13 answer
  with extra or fewer columns than the gold SQL selects (e.g. "Flower Awards | United States |
  2007" for "which festival"), 17 drop rows from a list, 3 add rows, 45 are wrong (wrong
  argmax / aggregate, wrong row picked; a few MMQA gold answers include columns the question does
  not ask for, e.g. "which store" → `doaktown | 28.74`), 1 empty. Throughput was **not**
  measured (all 400 answers came from a reader-cache shard of an earlier attempt in the same run
  folder) — smoke run: ~42k calls/h at batch 5.
- **Embedding caches pushed** (`mhtable-cache embeddings/bge-m3/`, code 8d359ee): spider_mmqa
  359,015 texts (4,982 questions, 757 DBs, 0.74 GB); bird 2,619,155 texts (5,511 questions,
  78 DBs, 5.4 GB — too big for laptop RAM/network, molab/Prajna only).
- **Open decision (reader ceiling):** options — prompt that asks for exactly the requested
  columns and *all* matching rows; a column-tolerant EM variant as a secondary metric; a larger
  reader (Qwen2.5-14B/32B fit in 96 GB, not on Kaggle T4s). Decide before P6/P9, since every
  arm is capped by this ceiling; P6's truncate baseline tells how much selection still adds.
- **Reader ablation results (molab 2026-11-19, `mhtable-runs/runs/reader-{7b,14b}-{p1,p2}-dev200`;
  oracle sub-tables, same 200 dev questions per tier, bf16).** The raw p2 numbers looked worse
  (7B strict 52% → 41%) — a **scoring artifact**: p2 makes the model end lines with a stray
  `|` ("Maurizio Felugo |"), which `parse_answer` read as an extra empty cell. Fixed 2026-11-20
  (blank cells at the edges of a line are separators; a deliberate empty cell is `""`). Re-scored
  from the saved answers (EM; paired bootstrap, 10,000 resamples, vs 7b-p1):

  | Arm | strict EM | relaxed EM | EM when answer shown (s / r) | vs 7b-p1 (strict / relaxed) | calls/h | load (warm) |
  |---|---|---|---|---|---|---|
  | 7b-p1 | **56.0%** | **50.5%** | 59.4% / 53.0% | — | ~50k | 50 s |
  | 7b-p2 | 54.0% | 49.0% | 57.7% / 51.8% | −2.0 [−7.5, +3.5] / −1.5 [−7, +4] | ~51k | 50 s |
  | 14b-p1 | 53.0% | 54.0% | 57.7% / 59.5% | −3.0 [−9.5, +3.5] / +3.5 [−3.5, +10] | ~41k | 136 s |
  | 14b-p2 | 50.0% | 50.5% | 53.7% / 54.2% | −6.0 [−13, +0.5] / 0.0 [−6.5, +6.5] | ~42k | 46 s |

  **No arm differs significantly from 7b-p1** (all 95% intervals contain 0). Neither the p2 rules
  nor the 14B reader raise the ceiling; the 14B is ~15–20% slower and loses Numerical questions
  (strict 26% vs 42%). **Decision proposal: keep Qwen2.5-7B + p1** (the report's reader,
  fastest, fits Kaggle T4s). The ~55% ceiling is a property of the task + gold labels (wrong
  argmax/aggregates, list completeness, MMQA gold answers with extra columns), so RL results are
  read as headroom closed relative to it (PLAN §5). The parser fix also lifts the P4 gate numbers
  (same 7b-p1 answers): strict 52.0% → 56.0%, relaxed 48.5% → 50.5%.
  **Throughput for planning** (batches ≤ 128, mean prompt ~580 tokens, answer ~20–26 tokens):
  7B ≈ 50k calls/h (≈ 8–10k prompt tokens/s), 14B ≈ 42k calls/h; model load 50 s warm (first
  download 2–4 min). A 300-iteration RL run (~24k calls) needs ≈ 30 min of reader time with 7B.
- **Reader ablation (code 2026-11-19):** prompt versions (`reader.prompt`: p1 = gate,
  p2 = exact requested attributes, every matching row, compare all rows; in the reader id and
  cache key), `evaluate --fresh` (no cached answers, so timing is real), reader load time and
  token throughput in `result.json`. Notebook `notebooks/reader_ablation.py` runs 7B/14B × p1/p2
  on the gate's 200 dev questions per tier (bf16, batches of 128 = one RL iteration) and
  tabulates EM by category plus calls/hour, prompt/answer tokens per second, load time.
- **Deferred:** embedding similarity for candidate ranking in the environment (changes the
  action set and every G-6 number); BC demonstrations at 1024-d need float16 or per-minibatch
  featurisation before the bge-m3 BC rerun (P7 note).
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

### P6 — Baselines + oracle ceiling (laptop with mock, then molab with the real reader) — CODE DONE 2026-11-21; real-reader run pending (`notebooks/track_c.py`)
- **Built:** `selectors/heuristic.py` (question–table cosine from the embedding cache; relevant
  tables = within δ = 0.1 of the best, ≤ 3; best anchor on them, else SEED; JOIN relevant tables
  by cosine; STOP), `TableEnv.materialise_tables` (B1 truncate: all tables, all columns, rowid
  order, cut at B; B2 random rows: equal budget share per table, seeded uniform sample),
  `evaluate --selector heuristic|truncate|random_rows` (whole-table baselines have no
  path_match / sufficiency); 111 tests pass.
- **Laptop dry run (data-v2 dev, full split; no real reader):** heuristic with the lexical `hash`
  encoder — join-path accuracy **8.4% strict / 7.2% relaxed** (BC with the same encoder: 31.5% /
  25.3%), answer shown 35% / 33%. B1 / B2 serialisation: **answer shown 93% / 92–93% at only
  ~720 / ~620 tokens, budget overflow 5% / 9%** — the Spider/MMQA dev databases mostly fit in
  4,096 tokens whole (G-2 again). So the P6 question "does selection beat reading everything?"
  hinges on whether the reader answers worse from whole databases (it must do the joins itself)
  than from the selected sub-table; selection's other gains are fewer tokens and the hard cases
  (BIRD, larger tables). The real-reader run settles it.
- **Real-reader results (molab 2026-11-21, `notebooks/track_c.py`, full dev 394 / 458, Qwen2.5-7B
  p1 bf16, data-v3) — P6 gate passed, P7 gate passed:**

  | Selector | EM strict / relaxed | set-F1 | path match | answer shown | tokens |
  |---|---|---|---|---|---|
  | oracle (ceiling) | 51.3% / 50.0% | 0.69 / 0.67 | 88.3% / 83.0% | 89% / 87% | 575 / 549 |
  | B1 truncate | 16.5% / 15.7% | 0.43 / 0.42 | — | 93% / 93% | 1,061 / 1,041 |
  | B2 random rows | 9.6% / 9.8% | 0.37 / 0.37 | — | 92% / 92% | 937 / 919 |
  | heuristic (bge-m3) | 21.1% / 20.1% | 0.36 / 0.36 | 22.1% / 18.8% | 55% / 53% | 470 / 436 |
  | **BC (bge-m3)** | **28.9% / 28.2%** | 0.45 / 0.45 | **44.9% / 38.4%** | 66% / 63% | 569 / 507 |

  **Selection matters:** whole tables contain the answer 93% of the time, yet the reader gets only
  16% right from them (it has to do the joins itself) vs 51% from the gold sub-table — the
  P4/P6 worry that B1 might rival the oracle is settled. **BC beats the heuristic** on join-path
  accuracy (44.9 vs 22.1 / 38.4 vs 18.8) and EM; bge-m3 lifted BC's dev path match from 31.5 /
  25.3 (hash) to 44.9 / 38.4, but the train–dev gap remains (train subset 84.7% / 78.7%). RL
  headroom on dev: BC 29% → ceiling 51% / 50%. BC trained in ~3 min per tier (best epoch 25 / 20).
  Reader throughput at full-dev batches: 60–90k calls/h.
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
- **2026-11-21:** demonstrations are stored in float16 (the policy casts to float32), which removes
  the 1024-d memory blocker; `configs/experiments/bc_bge.yaml` trains on the bge-m3 cache
  (`hf:embeddings/bge-m3/spider_mmqa.npz`, data-v3); `notebooks/track_c.py` runs it (strict +
  relaxed) and evaluates the checkpoints with the real reader next to the P6 baselines.
- **Open:** the gate needs the P6 heuristic selector to compare against; rerun BC with the real
  encoder once the P4 embedding cache exists. Memory note: demonstrations hold full feature
  arrays — fine at 64-d (≈ 0.2 GB), but with 1024-d embeddings ≈ 1–3 GB for Spider+MMQA(+BIRD):
  store float16 or featurise per minibatch before the bge-m3 runs.
- Original task list:
- Cross-entropy on canonical oracle trajectories; this is also the RL initialisation.
- **Gate:** BC beats the heuristic on dev join-path accuracy (mock reader first, then the real reader).

### P8 — PPO (laptop with mock → molab with the real reader) — CODE DONE 2026-11-17; gate G-8 FAILED (read 2026-11-18)
- **Built:** `algos/ppo.py`, the RL loop `train.run_rl` (curriculum, `--resume`, `train.jsonl` +
  `metrics.jsonl`), `selectors/oracle.py` (gold-path reference), candidate memoisation in
  `rollout.collect` (`cand_cache`), `configs/experiments/ppo.yaml`. 98 tests pass, including PPO
  overfitting the fixture question through the real entry point (G-8 in miniature).
- **First runs** (laptop CPU, data-v2 Spider+MMQA relaxed, `hash` encoder, mock reader, seed 0;
  a pipeline check, not a reportable result). G-8 passes if `train_pool_greedy.return` ≈
  `train_pool_gold_path.return`:

  | Run | Start | Pool / iterations | Greedy return (EM) | Gold-path return (EM) | Eval EM start → best | Eval path match | Reader calls | Minutes |
  |---|---|---|---|---|---|---|---|---|
  | `g8-r1` (R1) | random | 20 / 150 | 0.79 (80%) | 0.99 (100%) | 0% → 80% (train pool) | 55% | 1,442 | 32 |
  | `g8-r0` (R0) | random | 20 / 150 | 0.55 (55%) | 0.99 (100%) | 0% → 70% (train pool) | 35% | 1,308 | 31 |
  | `ppo-relaxed-r1-s0` | BC | 3,149 / 100 | 0.49 (51%) on 500 train | 0.81 (83%) | dev 37.3% → 47.8% (iter 40) | dev 25.3% → 28.0% | 8,059 | 25 |
  | `ppo-relaxed-r0-s0` | BC | 3,149 / 100 | 0.51 (53%) on 500 train | 0.81 (83%) | dev 37.3% → 48.7% (iter 100) | dev 25.3% → 26.9% | 8,044 | 24 |

- **Reading:** (1) **G-8 fails** — policy entropy collapses long before the pool is solved
  (R1: 1.84 → 0.26 by iteration 61 → 0.07; R0: → 0.04 by iteration 61 and ≈ 0 after), so PPO
  locks into wrong paths it never revisits; R0 also degrades late (sampled return 0.47 at
  iteration 136). R1 > R0 here, as shaping should give. (2) From BC, dev EM rises ~11 points
  but **join-path accuracy barely moves** and sub-tables grow (235 → 257 / 316 tokens): with the
  mock reader a sub-table that *contains* the gold rows counts as correct, so PPO gains by keeping
  more rows, and λ_c = 0.1 is too small to stop it. R0 and R1 are indistinguishable (one seed).
  This is a reward-design warning for the real reader, not yet a finding.
- **To fix G-8 (laptop only, mock reader; see §4b track A):** higher / non-decaying entropy
  coefficient, fewer PPO epochs or a smaller actor learning rate, an approximate-KL early stop,
  check the learnable temperature (it can sharpen the softmax on its own); rerun G-8 for R1 and R0.
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
- **Track B — code done 2026-11-21 (118 tests); molab run pending (`notebooks/track_b.py`).**
  Cause: a JOIN on an unindexed column makes SQLite build a temporary ("automatic") index inside
  every look-ahead query; an equality filter on an unindexed column scans the whole table.
  Laptop check on a synthetic 1M x 1M-row join: look-ahead COUNT 28 s -> 0.016 s with permanent
  indexes. Fix: (1) `env/sqlindex.py` builds indexes on join and anchor-able columns of tables
  with >= 50k rows (small DBs untouched, results unchanged) -> pushed as **`data-v4`** = data-v3 +
  indexed BIRD DBs; (2) env: look-ahead memo on (db, SQL), `materialise` reads huge / unknown-size
  relations unordered with a SQL LIMIT and returns an empty sub-table on timeout instead of
  crashing (the old path retried a 30 s sort and then raised, which coverage counted as a crash);
  sufficiency / entity-overlap skip relations of unknown size; (3) `analysis/speed.py` (per-DB
  timings + query plans) and `coverage.py --per-db` (BIRD instances are grouped by DB, so the old
  `--limit 500` only covered the first DBs). Small-relation output is byte-identical (Track C
  numbers stand). Notebook: download -> speed before -> index -> speed after -> coverage on 15
  questions per DB (train + bird_dev) -> push + tag data-v4 (separate buttons).
  **Still to check after the run:** look-ahead timeout (-1) rate on BIRD — the 2 s timeout is
  wall-clock, so -1 features are hardware-dependent; if the rate is not near 0, raise the timeout
  before P9 on BIRD.
- **Track B molab run (2026-11-21): gate passed, not yet pushed.** Indexes: 41 of 80 BIRD DBs
  changed, 31.6 -> 35.9 GiB, built in 2 min. Environment check with real SQL, 15 questions per DB:
  **0 crashes** in every tier/split, materialise timeouts 0; reachability matches the offline G-6
  (train 87.3% / 79.8% vs 87.4% / 79.5%; bird_dev 78.5% / 73.9% vs 80.1% / 73.6%) — the BIRD
  real-SQL check of §4a is now done. Look-ahead timeout (-1) rate over all DBs <= 0.9%. Speed on
  the 6 largest DBs (random episodes, 120 s each): mean look-ahead 0.23-0.66 s -> 0.07-0.31 s,
  2-7x more episodes; but random-episode timeouts there are still 2-12% (movie_platform 7%,
  talkingdata 8%, language_corpus 12%). The slowest plans are range or low-cardinality filters
  (`user_trialist < x`, `app_id > x`, `donation_total > x`) where SQLite, having no statistics,
  picks a weak index.
- **ANALYZE (2026-11-22) fixed most of it.** Random-episode look-ahead timeouts, indexes only ->
  + ANALYZE: movie_platform 7.3% -> 1.8%, talkingdata 7.9% -> 0.9%, bike_share_1 2.5% -> 0.5%,
  donor 1.9% -> 1.3%, codebase_comments 2.0% -> 1.9%; mean look-ahead 0.06-0.15 s; 2-4x more
  episodes per 120 s than with indexes alone (vs data-v3: 4-10x). **Outlier: language_corpus** (13.9%
  timeouts, 0.33 s mean): its slow queries join every row of `words` (millions) through an
  inferred edge that rarely matches, so the LIMIT never triggers; the -1 (unknown) flag covers
  those states. Accepted; revisit only if BIRD RL is slow there. Step-5 numbers above unchanged
  (ANALYZE changes speed only).
- **data-v4 tagged 2026-11-21** (verified on HF: 41 DB files with new hashes + manifest
  `stats/db_index_bird.json`, 80 BIRD DB files, data-v4 = main). **Track B done.**
  `eval.yaml` / `bc_bge.yaml` now pin data-v4 (Spider/MMQA files identical to data-v3).
- **BIRD relaxed coverage**, if it must reach 85% later: remaining blockers are values outside the
  10k cell index, non-inferred joins, join/numeric ranking, rare LIKE shapes; candidates are
  embedding similarity (P4) and a runtime exact-value probe.

## 4b. Road to P9 — three independent tracks (agreed 2026-11-18)

P9 (GRPO + the 2×2 main runs) starts once all three are done; they share no code paths and can
proceed in parallel. All figures are **run time (wall-clock execution) only**; coding is done
with an agent and is not the bottleneck. Estimates (2026-11-18) are rough: laptop CPU numbers
come from the P7/P8 runs; GPU numbers are typical figures for a 7B reader and bge-m3 until the P4
throughput notebook measures them.

| Track | Step | Laptop (16 CPU cores, RTX 4060 8 GB) | molab (RTX 6000 Pro 96 GB) |
|---|---|---|---|
| **A. G-8 fix** | sweep of ~8 G-8 runs (~30 min each, 4 in parallel) | 2–4 h | no gain: CPU-bound (featurise + 2.3M-param MLP) |
| | rerun PPO from BC, R1 + R0 in parallel | ~30 min | — |
| | **Track A total** | **≈ 2.5–4.5 h** | — |
| **B. BIRD look-ahead** | fetch BIRD DBs from HF | not possible (32 GB at ~370 KB/s ≈ 25 h; rule: no BIRD locally) | ~0.5 h |
| | build SQLite indexes on join/anchor columns | — | ~1 h |
| | push `data-v4` (~32 GB) | — | 0.5–1.5 h |
| | `analysis.coverage --family bird` | — | ~1 h |
| | **Track B total** (index route; a row-count cap alone needs only the fetch + check, ~1.5 h) | — | **≈ 3–5 h, one session; GPU unused** |
| **C. P4 + P6** | one-time setup: CUDA torch, 7B reader, bge-m3 (~10 GB) | 7–8 h download; 7B only in 4-bit; vLLM not native on Windows | ~10–15 min |
| | embedding cache: Spider+MMQA ≈ 214k texts | 10–20 min | 2–4 min |
| | embedding cache: BIRD, est. 1–2M texts | 1.5–3 h | 15–30 min |
| | P4 gate: oracle selector + reader, 200 dev questions | ~5 min | ~1 min + ~5 min vLLM start-up |
| | P6 baselines: 4 selectors × both tiers ≈ 3.4k reader calls | 1–2 h | ~5 min |
| | BC rerun with bge-m3, both tiers | 1–2 h | 40–80 min |
| | **Track C total** | **≈ 11–16 h** (half of it downloads) | **≈ 1.5–2.5 h** |

Reader throughput behind these: laptop ~2–3k calls/h (HF transformers, 4-bit); molab ~30–100k
calls/h (vLLM, bf16, batched).

**P9 preview** (measured: 100 PPO iterations × 128 questions ≈ 8k reader calls + ~25 min CPU):
one 300-iteration run ≈ 24k calls + ~75 min env/update → laptop ~10 h per run (12-run grid
{PPO, GRPO} × {R0, R1} × 3 seeds ≈ 5 days nonstop — impractical); molab ~1.5–2 h per run,
grid ≈ 20–25 h, i.e. 2–3 sessions run one at a time.

**Kaggle (2× T4, estimates):** reader ~5–15k calls/h (vLLM fp16, tensor-parallel over both T4s
or one 4-bit replica per T4; T4 has no bf16 and no FlashAttention-2); embedding cache ~5–10 min
(Spider+MMQA) / ~45–90 min (BIRD); P6 baselines ~15–40 min; one P9 run ≈ 3.5–7 h (reader 2–5 h +
env/update on 4 slower cores), so ~4–8 runs per weekly quota. **Fairness rule:** runs compared in
one table must use the same reader weights, precision and vLLM version on every platform, and
the reader `id` includes the precision so cache shards never mix. **Decision 2026-11-21: the
reader runs in bf16** (molab, Prajna A100/A40/L40S); Kaggle's T4s cannot, so Kaggle does no
reader runs that enter a comparison table (embeddings or other reader-free work only).

**Run time to reach P9:** on molab ≈ 2 h (C) + 3–5 h (B), plus ~3–4 h of A on the laptop in
parallel, i.e. about one molab day. Laptop-only cannot reach P9 (B needs molab; C would take
~11–16 h and the P9 grid ~5 days). With Kaggle in the mix: C runs on Kaggle while B runs on
molab (same ~one day, but in parallel); the P9 grid splits ~8 runs molab / ~4 runs Kaggle,
≈ 13–17 h of molab instead of 20–25 h, finishing within about a week of quota.

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
