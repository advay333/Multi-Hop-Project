# Design Document — Multi-Hop Table Reasoning via RL

Architectural view of the system: what the components are, what each one consumes and
produces, where data lives, and the rules that hold everywhere. For *when* each piece is
built, see `PLAN.md`. For the research framing (problem, MDP, rewards, metrics, algorithm
arms) see the report PDF; this document implements it.

---

## 1. What the system does

Given a natural-language question and a relational database (several tables linked by keys),
a small **learned policy** decides, one step at a time, which rows to filter and which tables
to join. **SQLite** executes each decision. When the policy stops, the resulting sub-table is
serialised under a token budget and handed to a **frozen LLM reader**, which answers. Training
uses RL (BC warm start, then PPO / GRPO and later other arms); the reward combines answer
correctness, a potential-based shaping term computed from the gold SQL, and a token cost.

```
question + database
        │
        ▼
 ┌──────────────┐  candidate actions   ┌──────────────┐
 │  TableEnv     │ ───────────────────► │   Policy     │  (only trained part, ~3M params)
 │ (SQLite, no   │ ◄─────────────────── │  π(a|s), V(s) │
 │  learning)    │     chosen action    └──────────────┘
 └──────┬───────┘
        │ STOP
        ▼
  materialise sub-table (≤ B tokens) ──► frozen Reader LLM ──► answer
                                                               │
                         training only: oracle + rewards ◄─────┘
```

## 2. Design principles

1. **Everything swappable is chosen by name in config** — reader, encoder, policy network,
   algorithm, selector, action tier, edge source, reward variant. Adding a new one is adding
   one class and one registry entry, never editing the loop.
2. **One rollout loop** for training and all evaluation modes; behaviour differs only via
   flags. Train/eval divergence is the most common source of wrong RL numbers.
3. **The environment is deterministic and gold-free.** All randomness is in the policy; gold
   information enters only through the oracle → rewards path (the firewall, §8).
4. **All data lives on HuggingFace.** Local disk is only a cache. Every artefact has one
   canonical location and a pinned revision.
5. **Notebooks contain no logic.** They install the package and run one entry point.
6. **Fair comparisons by construction:** every selector (learned, baseline, oracle) goes
   through the same materialiser, prompt, reader and budget.
7. **No database-specific integer IDs** in features or cache keys; everything is keyed by
   `table.column` strings so policies transfer to unseen databases.

## 3. Execution environments

| Environment | Role |
|---|---|
| **Laptop** (Windows, CPU, RTX 4060 8 GB) | Development, unit tests, Spider/MMQA builds, BC, PPO/GRPO smoke runs with the mock or a small HF reader |
| **molab** (cloud GPU, 12 h sessions, 90 min idle shutdown) | Data sourcing (downloads in the datacenter), BIRD builds, embedding cache, reader-in-the-loop RL, evaluation |
| **GitHub** `advay333/Multi-Hop-Project` (public) | Code |
| **HuggingFace** (private repos) | All data, caches, checkpoints and results (§4) |

Kaggle / Colab are interchangeable with molab: the same notebook body works anywhere that can
`pip install` the repo and read an `HF_TOKEN`.

## 4. Storage architecture

### 4.1 HuggingFace repositories

| Repo (private) | Type | Contents | Written by |
|---|---|---|---|
| `advay333/mhtable-raw` | dataset | Unmodified sources: `spider/` (release zip), `mmqa/` (the two mirror JSONs), `bird/` (train + dev zips), `CHECKSUMS.sha256`, `SOURCES.md` (URLs, dates) | `notebooks/source_data.py` (molab), once |
| `advay333/mhtable-data` | dataset | Built artefacts: `instances/`, `gold/`, `splits/`, `db/` (SQLite files), `index/` (edge candidates, cell index), `DATASET_CARD.md` | `mhtable.data.build` |
| `advay333/mhtable-cache` | dataset | `embeddings/<encoder>/` shards, `reader/<reader>/` answer-cache shards | `mhtable.encoders.build_cache`, reader cache flushes |
| `advay333/mhtable-runs` | model | `runs/<run_id>/` checkpoints, config, logs, metrics, result tables | `train.py`, `evaluate.py` |

Layout of `mhtable-data`:

```
instances/{spider,mmqa,bird}.jsonl        public fields only (§5.1)
gold/{spider,mmqa,bird}.jsonl             gold fields, keyed by instance_id (§5.2)
schemas/{spider,mmqa,bird}.json           db_ref -> SchemaGraph (tables, columns, PKs, FKs)
splits/<family>/{train,dev,test,bird_dev}.txt   instance ids
splits/<family>/db_assignment.json        database -> split (proves disjointness)
stats/<family>.json                       yields, tier counts, failure reasons, split sizes
db/spider/<db_id>.sqlite
db/mmqa/<schema_sig>.sqlite               materialised from MMQA's embedded tables
db/bird/<db_id>.sqlite
index/edges/<db_ref>.json                 (P3) candidate join edges + JAR features
index/cells/<db_ref>.jsonl                (P3) distinct (column, value) pairs for anchors
```

A **family** is a group of sources built and split together because their databases overlap:
`spider_mmqa` (MMQA reuses Spider databases) and `bird`. Families are built independently
(`python -m mhtable.data.build --family ...`) and pushed into the same repo.

### 4.2 Versioning

Each build is pushed as a commit and tagged (`data-v1`, `data-v2`, ...). Every config pins
`data.revision`; every run records the exact revisions of data, cache and code (git SHA) it
used. Results are reproducible from a run's `config.yaml` alone.

### 4.3 Local access — `mhtable.storage`

```python
fetch(repo: str, path_or_glob: str, revision: str | None = None) -> Path   # download into HF cache, return local path
push(repo: str, local_path: Path, path_in_repo: str, message: str) -> str  # returns commit sha
db_path(db_ref: str) -> Path       # "spider/concert_singer" -> local SQLite path (fetched on demand)
```

SQLite needs real files, so databases are fetched lazily per `db_ref` and kept in the HF cache.
The laptop only ever fetches what a job touches (e.g. Spider + MMQA DBs, ~1 GB); BIRD DBs are
fetched only on molab. The local `data/` directory is gitignored and can be deleted at any time.

### 4.4 Caches that are appended during runs

The reader cache is written by concurrent sessions, so it is **sharded**: each run writes its
own shard `reader/<reader>/<run_id>.jsonl` and loads all existing shards at start. A periodic
`merge` job compacts shards. Keys are `sha256(reader_id + prompt)`; values are the raw answer
string. The embedding cache is built once per encoder and is read-only afterwards.

## 5. Data model

All records are plain JSON-serialisable dataclasses in `mhtable/types.py`.

### 5.1 `Instance` — public, safe to show the policy and reader

| Field | Type | Meaning |
|---|---|---|
| `instance_id` | str | `"{source}:{db_id}:{n}"` |
| `source` | str | `spider` / `bird` / `mmqa` |
| `db_ref` | str | storage key of the SQLite file, e.g. `mmqa/3fa2c1` |
| `db_id` | str | source database (for splitting; never a feature) |
| `question` | str | natural-language question |
| `evidence` | str \| None | BIRD's external-knowledge hint, appended to the question |
| `tables` | list[str] | every table of the database (Spider/BIRD: the whole DB, so non-gold tables are natural distractors; MMQA: its 2–3 tables) |
| `n_gold_tables` | int | tables referenced by the gold SQL (reporting slices only) |
| `category` | str | Select / List / Count / Numerical |
| `tier` | dict | `{"strict": bool, "relaxed": bool}` |

### 5.2 `GoldRecord` — training signal only, stored in a separate file

| Field | Type | Meaning |
|---|---|---|
| `instance_id` | str | join key |
| `gold_sql` | str | as released |
| `gold_answer` | list[list[str]] | rows from executing `gold_sql`, each cell normalised with `norm_value` |
| `gold_tables` | list[str] | tables of the gold query, FROM/JOIN order |
| `gold_edges` | list[Edge] | join edges in the gold SQL (canonically ordered column pairs) |
| `gold_anchors` | list[Anchor] | filters `(table.column, op, raw literal values)`; compared via `Anchor.key()` |
| `gold_entity_sets` | list[dict] | after each gold action except STOP: `{"rows": n, "truncated": bool, "rowids": {table: [...]}}` (rowids capped at 2,000) |
| `gold_trajectory` | list[Action] | canonical action sequence for BC, with look-ahead row counts |
| `monotone` | bool | False for aggregation, ORDER BY…LIMIT, HAVING or negated filters (report A.7) |
| `fail_reasons` | list[str] | why it is outside a tier (empty if it fits), incl. `empty_answer`, `exec_error`, `exec_timeout`, `answer_too_large` |
| `reference_answer` | str \| None | answer text shipped by the source (MMQA only); kept because it often disagrees with executing the SQL (§11) |

Keeping gold in a separate file makes the firewall structural: code that only loads
`instances/` cannot leak gold.

### 5.3 Environment types

```python
Edge   = tuple[tuple[str, str], ...]          # ((t1.c1, t2.c2), ...) — composite keys allowed
Anchor = (column: str, op: str, values: tuple)  # "head.born_state", "=", ("Alabama",)
#   values are raw SQL literals (re-executable); Anchor.key() normalises them for comparison.
#   ops: strict "=", "<", ">"; relaxed adds "<=", ">=", "!=", "like", "not_like", "in", "not_in",
#   "between" (2 values), "is"/"not_is" (IS [NOT] NULL, no values)

@dataclass
class Action:
    kind: Literal["SEED", "ANCHOR", "JOIN", "STOP"]
    table: str | None
    anchor: Anchor | None
    edge: Edge | None
    lookahead_rows: int        # COUNT of the result if taken (cardinality estimate, gold-free)
    lookahead_tokens: int      # look-ahead fields are excluded from equality; Action.key() is
                               # the normalised identity used to match enumerated vs gold actions

@dataclass
class State:
    instance: Instance
    t: int
    visited: list[str]          # tables in the join tree, in visit order
    joins: list[Edge]
    anchors: list[Anchor]
    relation_sql: str           # SQL defining the current intermediate relation R_t
    n_rows: int
    est_tokens: int
```

### 5.4 Learning types

```python
@dataclass
class Step:
    state_feat: Tensor; action_feats: Tensor; mask: Tensor
    action_idx: int; logp: float; value: float
    reward: float = 0.0          # filled by rewards.assemble
    phi: float = 0.0             # potential, logged for diagnostics

@dataclass
class Trajectory:
    instance_id: str
    steps: list[Step]
    subtable: str; subtable_tokens: int
    answer: str | None
    reader_called: bool          # False on cache hit — counts against the call budget only when True
    info: dict                   # per-hop recall, join-path match, etc. (metric_only)
```

## 6. Modules

Package `src/mhtable/`. Each entry lists **demands** (inputs and dependencies) and **outputs**.

### 6.1 Infrastructure

**`config.py`** — dataclass tree loaded from YAML with CLI overrides (`key=value`).
Demands: a YAML path. Outputs: a frozen `Config`, saved verbatim into every run.

**`registry.py`** — `register(kind, name)` decorator and `build(kind, cfg)` factory for
`reader`, `encoder`, `policy`, `algo`, `selector`, `dataset`.

**`storage.py`** — §4.3.

**`types.py`** — §5.

### 6.2 Data layer (offline)

| Module | Demands | Outputs |
|---|---|---|
| `data/norm.py` | any value | `norm_value(v) -> str`: the only normaliser (casefold, whitespace, numeric canonical form, `<null>`, ISO dates) |
| `data/schema.py` | Spider/BIRD `tables.json` entry, or a SQLite file | `SchemaGraph`: `tables`, `columns`, `column_set(t)`, `is_pk(t,c)`, `fks`, `edges_from(t)`, `is_fk_edge(e)` |
| `data/sqlshape.py` | gold SQL + `SchemaGraph` | structural features, strict/relaxed failure reasons, question category |
| `data/execute.py` | SQLite path + SQL | read-only connection, `run(...)` with a wall-clock timeout, gold sub-table SQL |
| `data/raw.py` | raw repo paths | local file paths; zips extracted once into a checksum-keyed cache |
| `data/mmqa_db.py` | one MMQA record | SQLite file at `db/mmqa/<schema_sig>.sqlite` (types inferred, mojibake repaired); Spider `db_id` matches; keys borrowed from Spider |
| `data/loaders/{spider,bird,mmqa}.py` | extracted raw files | `SourceBundle`: `RawExample`s (question, SQL, db refs, audit fields), schemas, DB files |
| `data/split.py` | all `RawExample`s of a family | `{instance_id: split}`, `{database: split}`; asserts disjointness |
| `data/build.py` | `--family`, seed, horizon, timeout | everything in `mhtable-data` for that family (§4.1), pushed with `--push`, tagged with `--tag` |

### 6.3 Oracle

**`oracle.py`** — the only code that reads gold SQL. Called by the data build for every
question that fits the relaxed tier and has a non-empty answer.
Demands: `gold_sql`, `SchemaGraph`, an open read-only connection.
Outputs: `ParsedGold` (tables, edges, anchors, monotone) and an `OracleResult`
(trajectory, entity sets, failure reasons).

```python
parse(gold_sql, schema, horizon) -> (ParsedGold | None, features)   # None = outside relaxed tier
run_oracle(parsed, schema, con) -> OracleResult                     # trajectory, entity sets, checks
canonical_actions(parsed, start) -> list[Action]                    # see order below
items(edges, anchors) -> frozenset                                  # structural items for Φ
potential(taken, gold) -> float                                     # Jaccard; caller sets Φ(terminal)=0
```

**Canonical order:** start at the anchored table whose own filters leave the fewest rows; apply
its anchors; join remaining tables breadth-first, applying each new table's anchors right after
its JOIN (ANCHOR is only legal on visited tables); STOP. No anchors → SEED the first FROM table.

**Checks** (failures dropped and counted, never repaired): `oracle_reconstruct_mismatch` —
replaying the parsed plan does not give exactly the rows of the gold query's own
FROM/JOIN/WHERE; `join_not_key` — a gold join pairs two non-unique columns (JAR's
max(u_k, u_l) = 1 rule); `oracle_timeout` / `oracle_error`. Edge- and anchor-candidate recall
need the environment's candidate sets and are reported as **coverage** in P3 (gate G-6).

**`sqlrender.py`** — renders a state (visited tables, joins, anchors) as SQL with original
identifier spellings. Shared by the oracle and the environment so both define R_t identically.

### 6.4 Environment

| Module | Demands | Outputs |
|---|---|---|
| `env/edges.py` | `SchemaGraph`, DB | candidate edges with features (name similarity e, value Jaccard j, uniqueness u, score ω); `source: inferred \| gold_fk` |
| `env/cells.py` | DB | per-column distinct values (capped), ID-like columns excluded; `lookup(question) -> top-k (column, value)` |
| `env/actions.py` | `State`, edge and cell indexes, `tier` | `list[Action]` + mask, following the report's masking rules (H = 5, ≤ 2 or 3 anchors, no re-joins, STOP only for t ≥ 1) |
| `env/env.py` | DB path, indexes, config (`H`, `B`, `tier`, `distractors`) | `TableEnv` (below) |

```python
class TableEnv:
    def reset(self, inst: Instance) -> State
    def candidate_actions(self, s: State) -> tuple[list[Action], Tensor]   # actions, mask
    def step(self, s: State, a: Action) -> tuple[State, bool]              # next state, done
    def materialise(self, s: State, budget: int) -> tuple[str, int]        # markdown sub-table, tokens
```

`materialise` projects onto join keys, anchor columns, and the top-4 columns of visited tables
by schema similarity to the question; row order is deterministic; output is hard-truncated at
B tokens (with an overflow flag). The environment never imports `oracle` or reads `gold/`.

### 6.5 Models

**`encoders/`** — `Encoder.encode(texts: list[str]) -> np.ndarray[n, d]`.
`build_cache` embeds every question, `table: col1, col2, ...` string, column name and indexed
cell value once (molab GPU) and pushes shards to `mhtable-cache`. `EmbCache.get(text)` is a
dict lookup; a cache miss is an error at training time, not a silent re-encode.

**`featurise.py`** — Demands: `State`, `list[Action]`, `EmbCache`, flags (`header_ablation`).
Outputs: `state_feat [d_s]`, `action_feats [|A|, d_a]`. Asserts no gold field is reachable.
Count-like scalars are `log1p`-transformed.

**`policies/`** —
```python
class Policy(nn.Module):
    def forward(self, state_feat, action_feats, mask) -> tuple[Tensor, Tensor]  # logits [B,|A|], value [B]
```
`mlp` (default): state tower + action tower → d = 256, scaled dot product with learnable
temperature, separate value head. `transformer` (later): same interface, state encoded as
tokens (question, visited tables, anchors, scalars).

**`readers/`** —
```python
class Reader:
    id: str                                                        # part of every cache key
    def read_batch(self, questions: list[str], subtables: list[str]) -> list[str]
```
Implementations: `mock` (answers correctly iff the sub-table is sufficient — needs gold, so
allowed only in training/debug and labelled as such), `hf` (transformers, any size, laptop
smoke tests), `vllm` (molab). A `CachedReader` wrapper adds batching, the sharded cache and a
call counter. The prompt template lives in `readers/prompt.py` and is shared by all readers.

### 6.6 Learning and evaluation

**`rewards.py`** — Demands: `Trajectory`, `GoldRecord`, reader answer, config (`variant`:
`R0` sparse | `R1` shaped, `beta`, `lambda_cost`). Outputs: per-step rewards written into
`Step.reward`, and `phi` for logging.
`r_t = β[Φ(s_{t+1}) − Φ(s_t)] + 1[a_t = STOP]·(R_ans − λ_c·tok/B)`.

**`rollout.py`** — the shared collector.
```python
def collect(env, actor, reader, instances, *,
            mode: Literal["sample", "argmax"],
            oracle_mode: Literal["reward", "metric_only", "off"],
            group_size: int = 1,              # >1 for GRPO / MC value estimates
            start_after_gold_steps: int = 0,  # curriculum
            ) -> list[Trajectory]
```
`actor` is either a `Policy` (via featurise) or a `Selector`. Steps all live episodes in
lockstep, batches the reader call at the end, then (if `oracle_mode="reward"`) calls
`rewards.assemble`. `oracle_mode="reward"` is rejected when called from `evaluate.py`.

**`algos/`** —
```python
class Algorithm:
    def compute_advantages(self, trajs: list[Trajectory]) -> Batch      # advantages, returns
    def update(self, batch: Batch) -> dict                              # stats: loss, entropy, KL, explained var
```
| Arm | Advantage | Update |
|---|---|---|
| `bc` | — (targets are gold actions) | cross-entropy |
| `ppo` | GAE(γ = 1, λ = 0.95) with learned V | clipped ratio, value clipping, entropy, KL to BC |
| `grpo` | group-normalised return over G rollouts, applied to every step | clipped ratio + KL to BC; logs zero-advantage fraction |
| later: `reinforce`, `ppo_kl`, `vineppo` | batch mean / same as PPO / MC values from K resets | one step / adaptive KL / clipped ratio |

**`selectors/`** — non-learned actors with the policy's interface:
`truncate`, `random_rows`, `heuristic` (greedy cosine), `oracle` (gold trajectory — eval
ceiling only), later `prompted_planner`, `sql_extract`.

**`metrics.py`** — Demands: trajectories + gold. Outputs: EM (normalised), set-F1, join-path
exact accuracy, sub-table answer coverage, edge P/R, per-hop recall, reader tokens; sliced by
source, 2-/3-table, category, monotone vs non-monotone. Paired bootstrap for differences.

**Entry points** —
`python -m mhtable.data.build --config configs/data.yaml`
`python -m mhtable.encoders.build_cache --config ...`
`python -m mhtable.train --config configs/experiments/ppo_r1.yaml`
`python -m mhtable.evaluate --config ... --selector policy --run <run_id>`

## 7. End-to-end flows

**Sourcing (once, molab):** download Spider (Google Drive), MMQA mirror (Google Drive), BIRD
(official zips) → checksum → push to `mhtable-raw`.

**Build (molab, `notebooks/build_data.py`; laptop only for `--limit` dev runs):** fetch raw →
loaders → materialise MMQA DBs → execute gold SQL + tier analysis → split → (P2) oracle labels →
(P3) edge and cell indexes → push `mhtable-data`, tag `data-vN`.

**Embed (molab):** fetch `mhtable-data` → collect all texts → encode → push `mhtable-cache`.

**Train (laptop smoke / molab real):**
```
fetch data + cache → build env, policy, reader, algo from config
for iteration:
    trajs = rollout.collect(env, policy, reader, batch, mode="sample", oracle_mode="reward")
    batch = algo.compute_advantages(trajs)
    for epoch: stats = algo.update(batch)
    log stats (+ reader calls so far); every k iters: evaluate on dev slice
    checkpoint → push mhtable-runs (resumable)
```

**Evaluate:** `rollout.collect(mode="argmax", oracle_mode="metric_only")` → `metrics` → result
table pushed to the run folder.

## 8. Invariants (asserted in code)

1. **Firewall:** `featurise`, `env`, `readers/prompt.py` never receive a `GoldRecord`;
   `evaluate.py` refuses `oracle_mode="reward"`; the `mock` reader and `oracle` selector are
   refused in any run tagged as a reported result.
2. **Split disjointness:** no `db_id` appears in two splits (checked at build and at load).
3. **One normaliser:** answers, entity sets and anchors all pass through `norm_value`.
4. **Shaping telescopes:** per episode, Σ_t β[Φ(s_{t+1}) − Φ(s_t)] = 0 (unit-tested).
5. **Determinism:** env transitions and materialisation are pure functions of (state, action).
6. **Matched budget:** every comparison reports reader calls (cache misses) consumed.

## 9. Extension points

| To add | Touch |
|---|---|
| an RL arm | one file in `algos/` + registry entry + config |
| a reader | one file in `readers/` (`id` + `read_batch`) |
| a policy network | one file in `policies/` with the same `forward` |
| a baseline | one file in `selectors/` |
| a dataset | one loader in `data/loaders/` producing `(Instance, gold_sql)`; build does the rest |
| an action type | `env/actions.py` + its featurisation; oracle mapping in `canonical_trajectory` |

## 10. Testing

`pytest` on CPU in under two minutes, using a tiny fixture database (the `head / management /
department` example) committed under `tests/fixtures/`:
norm round-trips; oracle entity sets on known queries; env masking and determinism; the gold
trajectory reaches the gold sub-table; shaping telescopes; GAE at γ = 1; firewall assertions;
a 20-question PPO overfit with the mock reader (gate G-8, marked slow).

## 11. Known data caveats

- **MMQA's answer strings often disagree with its own SQL.** Over the 3,114 questions whose
  SQL runs: 65% agree once formatting is ignored (sentences, pandas-dict answers, joined or
  dropped columns, rounding); 17% partly agree (MMQA lists fewer rows or picks one of several
  tied rows); 18% genuinely differ (wrong counts, different values, or the SQL returns nothing).
  The original Spider databases give the same SQL results as our materialised tables 99% of the
  time, so the disagreement is in MMQA's annotations, not our reconstruction. `gold_answer` is
  the SQL execution (consistent with the oracle and rewards); `reference_answer` keeps the
  shipped text. Consequence: our MMQA scores are not directly comparable to MMQA's published
  numbers, which were scored against the shipped answers.
- **6% of MMQA gold SQL does not run** (missing tables, MySQL functions, typos) and 3% returns
  nothing; these are excluded via `fail_reasons`.
- **MMQA's embedded tables contain mojibake** ("MisÃ©rables"); repaired at materialisation.
- **15% of Spider questions return an empty answer** on Spider's own databases; excluded.

## 12. Glossary

| Term | Meaning |
|---|---|
| Table / row / column | A spreadsheet-like grid; one record; one field |
| Primary key (PK) | Column that uniquely identifies each row of its table |
| Foreign key (FK) | Column holding another table's PK — the links between tables |
| Join / edge | Combining two tables where linked key values match; an edge is one such link |
| Anchor | A filter `column op value`, e.g. `born_state = 'Alabama'` |
| Hop | One join step from a table to the next |
| Gold SQL / answer | The released correct query and its result |
| Entity set E_t | Rows (keys) consistent with all decisions so far |
| Sub-table | What the reader finally sees, ≤ B tokens |
| Reader | Frozen LLM that answers from the sub-table |
| Oracle | Code that uses gold SQL to compute training rewards and diagnostics |
| Tier | `strict` (report's action set) or `relaxed` (superset with LIKE/IN/BETWEEN/etc.) |
| R0 / R1 | Sparse reward / reward with potential-based shaping |
