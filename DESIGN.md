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
| `advay333/mhtable-cache` | dataset | `embeddings/<encoder>/<family>.npz` + `<family>.json` manifest, `reader/<reader>/` answer-cache shards | `mhtable.encoders.build_cache`, reader cache flushes |
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
| `gold_entity_sets` | list[dict] | after each gold action except STOP: `{"rows": n, "truncated": bool, "rowids": {table: [...]}}` (rowids capped at 2,000; a `WITHOUT ROWID` table lists its primary key as a JSON array string, e.g. `"[1,\"ca\"]"`) |
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
FROM/JOIN/WHERE; `join_not_key` — a gold join pairs two non-unique keys (JAR's
max(u_k, u_l) = 1 rule; a composite join `a.x = b.x AND a.y = b.y` is tested on each side's
column combination, from data-v3); `gold_relation_empty` — the gold FROM/JOIN/WHERE selects no rows
(added in P3; data-v2 predates it, so loaders apply the same rule); `oracle_timeout` /
`oracle_error` (oracle queries get their own, longer timeout: `--oracle-timeout`, default
120 s). Entity sets identify rows by rowid; `WITHOUT ROWID` tables (BIRD `language_corpus`) use
their primary key as a JSON array string instead (from data-v3). Gold analysis (`sqlshape`)
rejects LEFT/RIGHT/FULL joins (`outer_join`) and records `col LIKE 'text'` without wildcards as
`col = 'text'` (from data-v3; the reconstruction check catches any case-sensitivity difference).
Edge- and anchor-candidate recall
need the environment's candidate sets and are reported as **coverage** in P3 (gate G-6).

**`sqlrender.py`** — renders a state (visited tables, joins, anchors) as SQL with original
identifier spellings. Shared by the oracle and the environment so both define R_t identically.

### 6.4 Environment

| Module | Demands | Outputs |
|---|---|---|
| `env/text.py` | strings | tokens (camel/snake split, stopwords dropped, plurals folded), numbers incl. number words/ordinals, ISO dates, quoted phrases, operator cues; `Similarity` interface with `LexicalSimilarity` (the P4 embedding scorer implements the same interface) |
| `env/index.py` | `SchemaGraph`, DB | `DbIndex` per database, stored as `index/<db_ref>.json`: column stats (rows, distinct, uniqueness u, numeric, min/max, avg length); candidate joins with JAR features (e name similarity — identically named text columns get at least 0.5, since `name` is a stopword — j value containment, u, ω = (e + j)·max u, halved for id-to-id pairs); cell index (≤10,000 most frequent values per non-ID column). `CellSearch.lookup(text)` → top (column, value, score) |
| `env/actions.py` | `State`, `EpisodeContext`, `ActionConfig` (tier, edges mode, threshold, caps) | legal candidate actions under the masking rules |
| `env/env.py` | `EnvConfig`, `EnvData` (local repo copy or HF) | `TableEnv` (below) |
| `env/sqlindex.py` | local real-file copy of `db/`, `schemas/`, `index/` | SQLite indexes in place on large tables (≥ 50k rows): both sides of candidate joins (ω ≥ 0.5 or declared; composite edges → one composite index) and anchor-able columns (cell-index or numeric, avg ≤ 64 chars); skips columns an index already leads; small DBs untouched; `db_index_manifest.json`. Changes speed only, never results (Track B, data-v4) |

```python
class TableEnv:
    def reset(self, inst: Instance, tables: list[str] | None = None) -> State
    def candidate_actions(self, s: State) -> list[Action]   # only legal actions; each with look-ahead
    def step(self, s: State, a: Action) -> tuple[State, bool]   # next state, done
    def materialise(self, s: State, budget: int | None = None) -> Materialised  # text, tokens, rows, overflow
```

**Masking** (illegal actions are never offered): t = 0 → SEED or ANCHOR (an anchor seeds its
table); t ≥ 1 → ANCHOR on visited tables (≤ 2 strict, ≤ 3 relaxed, no duplicates), JOIN to an
unvisited episode table, STOP; t = H − 1 → STOP only. Caps: ≤ 6 SEED, ≤ 14 ANCHOR, 1 STOP;
JOIN gets ≥ 6 slots plus whatever the anchors leave unused, at most 12, so |A(s)| ≤ 25. Joins
rank by ω + 2 · sim(question, target table).

**Anchor candidates** come from the question (+ BIRD evidence) and the database only:
explicit filters written out in the text (BIRD evidence: `WasCompiled = 0`, `IndepYear BETWEEN
1980 AND 1995`, `Name LIKE '%x%'`, `IS NOT NULL`), resolved to episode columns and ranked first
(strict keeps only =, <, >); categorical `=` from cell-index matches (exact phrase, token and
5-letter-prefix matching); numbers/number words × the 3 most plausible numeric columns
(named in an explicit filter, value range contains the number, then name similarity) plus, per
number, up to 2 columns named by the words around it ("game ID 178", "254 likes") ×
operators (=, <, > always; ≤, ≥, ≠ when the wording cues them); dates and years against
date-like text columns; relaxed adds BETWEEN, IN, ≠ on matched values under negation, LIKE for
quoted phrases / "the letter x" / partial matches / date prefixes (`release_date LIKE '2006%'`,
"February 2017" → `'2017-02%'`).

**Look-ahead:** every candidate carries the capped COUNT(*) of the relation it leads to and an
estimated token count — cardinality estimation, not gold information. A count that exceeds
`lookahead_timeout_s` (2 s, wall-clock) is reported as −1 (unknown; a feature flag). Counts are
memoised per env on (db_ref, SQL) — the count depends only on the relation, so it is computed
once per process (`lookahead_memo`, cleared when full). `TableEnv.stats` / `slow_queries` record
timings for `analysis/speed.py`. On BIRD the speed depends on the SQLite indexes of data-v4
(`env/sqlindex.py`): an unindexed JOIN makes SQLite build a temporary index inside every query.

`materialise` projects onto join keys, anchor columns and the other columns of the visited
tables ranked by question relevance — as many as let all rows fit B (estimated), at least 4
(`project_all_columns`; off = exactly the top 4, the pre-2026-11-18 behaviour) — orders rows deterministically by rowid, and cuts rows to fit B tokens
(`overflow` flag). Relations at the look-ahead cap (≥ 100k rows) or of unknown size are read
without ORDER BY with a SQL LIMIT (scan order; deterministic for a fixed DB file, its indexes and
SQLite build) so SQLite never sorts millions of joined rows; a fetch timeout
(`fetch_timeout_s`, 30 s) gives an empty sub-table with `timed_out=True` instead of a crash. The budget is counted by `env/tokens.py`: 3.5 chars/token by default
(`env.tokenizer: null`, mock-reader runs), or the reader's own tokenizer (`env.tokenizer: reader`
or a model name; rows are counted in blocks, then the joined text is re-counted so the result is
exact). Look-ahead token estimates stay character-based. The environment never imports the
oracle or reads `gold/` (enforced by a test).

**Coverage analysis** (`analysis/coverage.py`, gold-aware, outside the environment) follows each
gold trajectory through the environment and reports gate G-6, per-step coverage, reasons for
misses, the edge threshold for 98% gold-join recall, and random-policy smoke tests (needs the
databases). `analysis/reach.py` measures the same reachability **without databases** —
membership in the candidate set depends only on question, schema and index — so BIRD coverage
runs on the laptop; it also reports *sole blockers* (questions whose only missing reason is X)
and |A(s)|. Its numbers matched `coverage.py` exactly on data-v2. `coverage.py --per-db N`
samples N questions from every database (instance files are grouped by DB, so `--limit` alone
only reaches the first few). `analysis/speed.py` profiles look-ahead / materialise time per
database with random episodes and prints SQLite's query plan for the slowest look-aheads.

### 6.5 Models

**`encoders/`** — `Encoder.encode(texts: list[str]) -> np.ndarray[n, d]`; `hash` (tests) and
`st` (sentence-transformers, `BAAI/bge-m3`, 1024-d, built in P4).
`python -m mhtable.encoders.build_cache --family F --revision data-vN [--push]` embeds exactly
`featurise.required_texts` (with and without header ablation) for every question of the family
in either tier and every split — it reads `instances/`, `schemas/`, `index/`, never `gold/` —
in resumable 50k-text parts, then writes `embeddings/<encoder>/<family>.npz` (float16) and a
`<family>.json` manifest to `mhtable-cache`. `EmbCache.load(paths)` keeps one float16 matrix and
a text → row dict; `get` returns float32; a miss on a required text is an error at training
time, not a silent re-encode. Configs point at it with `encoder.cache: hf:embeddings/bge-m3/<family>.npz`
(or a list).

**`featurise.py`** (built in P5) — `Featuriser(env, cache, FeatConfig)(state, actions)` →
`state_feat [d_s]`, `action_feats [|A|, d_a]` (numpy). Each vector is a fixed number of
embedding slots followed by scalars scaled to about [0, 1] (counts through `log1p`):

| Vector | Embedding slots (each `d_emb`) | Scalars |
|---|---|---|
| state | question; mean of visited tables; mean of applied filters (column + value) | 6: t/H, rows, tokens/B, #tables, #filters, rows-unknown flag |
| action | what it acts on (table / filter column / mean of the join's columns); filter value; target table | 25: action kind (4), operator class (5), cosine to the question (2), name-similarity scores (2), join ω, e, j, u + unscored/composite flags (6), look-ahead rows, tokens, unknown flag, change in rows (4), value flags (2) |

Sets are mean-pooled (order-free). Texts looked up in the cache are produced by
`question_text`, `table_text`, `column_text`, `value_text`; `required_texts(inst, schema, index)`
lists them all for the P4 cache builder. Filter values missing from the cache (question
numbers, LIKE patterns) become a zero vector plus a flag. `header_ablation` renames tables and
columns to `table_i` / `col_j` and zeroes the name-similarity scalars (policy inputs only —
candidate generation still reads real names). `firewall.assert_gold_free` runs on every input,
and a test checks the module imports nothing gold-aware.

**`policies/`** —
```python
class Policy(nn.Module):
    def forward(self, state_feat, action_feats, mask) -> tuple[Tensor, Tensor]  # logits [B,|A|], value [B]
```
`mlp` (default, built in P5; 2.3M parameters at d = 256 with 64-d embeddings): state tower +
action tower → d = 256, scaled dot product with learnable temperature; the critic has its own
state tower. Each tower projects the embedding slots with one shared linear layer + LayerNorm
and sends the scalars through a separate MLP. Built as `MLPPolicy(featuriser.spec, d, hidden)`.
Padded candidates get logit −1e9. `transformer` (later): same interface, state encoded as
tokens (question, visited tables, anchors, scalars).

**`readers/`** —
```python
class Reader:
    id: str                                                        # part of every cache key
    def read_batch(self, questions: list[str], subtables: list[str], meta=None) -> list[str]
```
Implementations: `mock` (built in P5: answers correctly iff the sub-table is sufficient — needs
gold through `meta`, so allowed only in training/debug, and refused by the rollout when
`oracle_mode="off"`), `vllm` (P4; the main reader: Qwen2.5-7B-Instruct, float16 on every
platform, greedy) and `hf` (P4; transformers, laptop smoke tests and the fallback when vLLM will
not install). Both share `readers/llm.py`: the prompt wrapped in the model's chat template, rows
cut from the end if the prompt would exceed the context window (counted), answer clean-up
(fences, "Answer:", bullets), `ReaderStats` (prompts, cut prompts, answers cut by
`max_new_tokens`, seconds, calls/hour) and `reader_id` = backend : model : precision : greedy :
max tokens : prompt version, so caches from different setups never mix. Prompt versions live in
`readers/prompt.py` (`reader.prompt`: `p1` original, `p2` explicit column/row rules). Each accepts and ignores
the other's backend-only keys, so one config serves both.
`CachedReader` (built in P5) adds the cache keyed by `sha256(reader id + prompt)`, the optional
shard file and the call counter (`calls` = cache misses; identical prompts in one batch are sent
once); it returns `(answers, called)`. The prompt template lives in `readers/prompt.py`, takes
strings only, and is shared by all readers. Real readers ignore `meta`; it never enters the prompt.

**`sufficiency.py`** (gold-aware, built in P5) — `path_match(state, gold)`: the policy's joins
and filters equal the gold query's. `sufficient(env, state, materialised, gold)`: the sub-table
holds what the answer needs — the gold path always; otherwise the oracle's final entity set is
compared by row id per table (monotone questions: gold rows ⊆ policy rows; non-monotone: same
tables, same row count, same row ids). Sub-tables cut by the budget, truncated gold entity sets
and oversized relations count as insufficient. It does not check which columns are shown, so it
is the mock reader's rule and a diagnostic, not a reported metric.

**`metrics.py`** — the reader is asked for one answer row per line with cells separated by
` | `. `em_norm` is 1 when the predicted rows equal the gold rows as sets after `norm_value` (row
order free), **or** when the two contain the same set of values regardless of row structure (a
reader that lists values one per line is not punished). This defines the outcome reward R_ans.
Empty cells are written `""` and a `|` inside a value as `/`. P4 adds the rest of report
Table 3: `set_f1`; `answer_in_subtable` (share of gold answer values that are cells of the
sub-table the reader saw — separates projection/budget loss from reader errors; read by
category, since Count/Numerical answers are computed); `item_counts` (join-edge and filter
precision/recall, micro-averaged); per-hop entity recall/precision vs E* (`sufficiency.entity_overlap`,
hop = position of the table in the gold trajectory); `summarise` (overall and by source,
2-/3-table, category, monotone; also `em_when_answer_shown`); `paired_bootstrap` (10,000
resamples, 95% interval, p). The rollout computes the per-question extras only with `detail=True`
(evaluation), since entity overlap costs one SQL query per episode.

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
            golds: dict[str, GoldRecord] | None = None,   # required unless oracle_mode="off"
            group_size: int = 1,              # >1 for GRPO / MC value estimates
            start_after_gold_steps: int = 0,  # curriculum (training only)
            reward_cfg: RewardConfig | None = None, generator=None,
            ) -> list[Trajectory]             # rollouts of instances[i] are consecutive
```
`actor` is anything with `act(states, candidate_lists, mode, generator) -> list[Decision]`:
`PolicyActor(policy, featuriser)` or a P6 selector. Steps all live episodes in lockstep (one
batched forward per step), batches the reader call at the end (empty sub-tables are not sent),
fills `traj.info` (`sufficient`, `path_match`, `r_ans`, `visited`, `overflow`, ...) when gold is
available, then (if `oracle_mode="reward"`) calls `rewards.assemble`. An episode started after
k gold steps has shaping terms summing to −β·Φ(s_k) instead of 0. `oracle_mode="reward"` is
rejected when called from `evaluate.py`.

**`algos/`** —
```python
class Algorithm:
    def compute_advantages(self, trajs: list[Trajectory]) -> Batch      # advantages, returns
    def update(self, batch: Batch) -> dict                              # stats: loss, entropy, KL, explained var
```
`algos/base.py` (built in P5) also holds what every arm shares: `Batch` (all steps flattened,
candidates padded, with `traj` and `group` indices), `collate(trajs)`, `reward_to_go`,
`gae(rewards, values, gamma, lam)` with V(terminal) = 0, and `evaluate_actions(policy, batch)`
(log-probability, entropy, value, full log-probabilities for KL terms).
| Arm | Advantage | Update |
|---|---|---|
| `bc` | — (targets are gold actions) | cross-entropy |
| `ppo` | GAE(γ = 1, λ = 0.95) with learned V | clipped ratio, value clipping, entropy, KL to BC |
| `grpo` | group-normalised return over G rollouts, applied to every step (`advantage: outcome`); or per-step λ-return minus the group mean at the same step index (`advantage: process`) | clipped ratio + KL to BC; logs zero-advantage fraction |
| later: `reinforce`, `ppo_kl`, `vineppo` | batch mean / same as PPO / MC values from K resets | one step / adaptive KL / clipped ratio |

**`dataset.py`** (gold-aware, built in P7) — `load_split(EnvData, split, tier, sources, limit, seed)`
→ `(Instance, GoldRecord)` pairs: in the split, inside the tier, with usable oracle labels;
reads a local copy of the data repo or HuggingFace.

**`algos/bc.py`** (built in P7) — `demonstrations(env, featuriser, items)` follows each gold
trajectory through the environment and records features plus the index of the gold action
among the candidates; questions with a gold action the environment does not offer give no
demonstration and are counted. `BC.update` is one epoch of cross-entropy over minibatches
(AdamW, gradient clipping, optional label smoothing) on the **actor only**; `BC.evaluate` gives
teacher-forced loss, per-step accuracy and whole-trajectory accuracy.

**`algos/ppo.py`** (built in P8) — `compute_advantages`: GAE per trajectory with the stored
V(s), V(terminal) = 0. `update`: K epochs of minibatches on clipped ratio + clipped value loss
− entropy + KL(π ‖ π_BC) against a frozen copy of the BC policy (term dropped when there is no
BC start); separate actor/critic learning rates; advantages normalised per batch; entropy
coefficient annealed by `set_progress`. Stats: policy/value loss, entropy, KL to BC, approximate
KL to the previous policy, clip fraction, explained variance of the critic. Stability options
(Track A, all off by default): `target_kl` (stop an update's epochs once a minibatch's approximate
KL exceeds 1.5× the target), `freeze_temperature` (the softmax temperature is not learned),
`adam_eps`. Measured (PLAN P8): one Adam step at actor lr 3e-4 moves a converged policy by
KL ≈ 2.6, because Adam steps each of the 2.3M parameters by about lr whatever the gradient's size.

**`algos/grpo.py`** (built 2026-11-22, P9) — a subclass of PPO with no critic.
`compute_advantages`: trajectories come in consecutive blocks of `group_size` (= G, default 8)
rollouts of one question from one start (as `collect` returns them; a question sampled twice is
two groups); each trajectory's total return R_i becomes A_i = (R_i − mean) / (std + ε) of its
group (`std_normalize: false` = mean-only, "Dr. GRPO" — the default in grpo.yaml, see PLAN P9),
the same A_i for every one of its steps. Because shaping telescopes over an episode, outcome
GRPO's advantage ignores it: R1 ≡ R0 exactly, because all G rollouts of a group share the start state (even under the curriculum), so the shaping sum −β·Φ(start) is a per-group constant that the mean removes. Corrected 2026-11-23; earlier text said "up to the curriculum start".
**`advantage: process`** (2026-11-23, P9 ablation; `process_advantages()`): per step, raw_t = Σ_k λ^k r_{t+k}, which is GAE(γ 1, `process_lam`) with V ≡ 0. The baseline is the mean raw value of the group's rollouts still running at step index t, never a function of the rollout's own length. The shaping only matters for λ < 1: at λ = 1 it reduces to −β·Φ(s_t), a state-only term. PPO is limited the same way (γ 1, λ 0.95): its R1 signal comes only from GAE truncation and the critic's bootstrapping. The DeepSeekMath form (sum of pooled-normalised step rewards) was rejected: the shaping still cancels, and the pooled mean adds a per-step length penalty. Config `grpo_process_real.yaml`, script `slurm/p9_process.sh`;
a group with identical returns gets A = 0 and counts towards `zero_adv_frac`. `update`: PPO's
clipped surrogate, entropy and KL to BC, with `value_coef = 0` and an actor-only optimiser (the
value head never changes). `configs/experiments/grpo.yaml` uses 16 questions × 8 rollouts = 128
episodes per iteration, PPO's 128 × 1, so both arms spend the same reader calls per iteration.

**RL loop** (`train.run_rl`, built in P8; report Algorithm 1) — per iteration: sample
`questions_per_iter` training questions (a smaller pool is repeated to fill the batch); with
probability p (annealed to 0) a question starts after k gold steps; `rollout.collect(sample,
reward, group_size=algo.group_size)`; `algo.compute_advantages` → `algo.update`. Logs one row
per iteration to `train.jsonl` (return, EM, path match, entropy, KL, reader calls) and a dev (or
training-pool) evaluation every `eval_every` iterations to `metrics.jsonl`; keeps the best
checkpoint by eval EM and `last.pt` for `--resume`. At the end it reports two reference
numbers on the training pool: the gold path's return (`selectors/oracle.py`) and the final
greedy policy's. Candidate enumerations are memoised across the whole run (`cand_cache`), since
the environment is deterministic — group members and repeated questions cost no extra SQL.

**`train.py`** (built in P7 for `algo.name: bc`, P8 for the RL arms) — builds env,
featuriser, policy, reader and algorithm from the config, trains, evaluates on dev with the
shared rollout (`argmax`, `metric_only`), keeps the checkpoint with the best dev join-path
accuracy, and writes `<out>/<run_id>/{config.yaml, metrics.jsonl, policy.pt, result.json}`
(local `runs/`, gitignored; nothing is uploaded). `load_checkpoint(path)` rebuilds the policy —
the RL arms start from it. `evaluate_rollouts` reports path match, sufficiency, EM, overflow,
steps and tokens, overall and by source and number of gold tables.

**`selectors/`** — non-learned actors with the policy's interface: `heuristic` (built in P6:
question–table cosine from the embedding cache, relevant tables within δ of the best, best
anchor, then JOIN relevant tables, STOP) and `oracle` (gold trajectory — eval ceiling only).
The whole-table baselines B1 `truncate` and B2 `random_rows` have no action path: they use
`TableEnv.materialise_tables(inst, order="rowid"|"random")` (all tables and columns, same token
counter and budget) and are run by `evaluate.py`. Later: `prompted_planner`, `sql_extract`.

**`metrics.py`** — Demands: trajectories + gold. Outputs: EM (normalised), set-F1, join-path
exact accuracy, sub-table answer coverage, edge P/R, per-hop recall, reader tokens; sliced by
source, 2-/3-table, category, monotone vs non-monotone. Paired bootstrap for differences.

**Entry points** —
`python -m mhtable.data.build --config configs/data.yaml`
`python -m mhtable.encoders.build_cache --config ...`
`python -m mhtable.train --config configs/experiments/ppo_r1.yaml`
`python -m mhtable.evaluate --config configs/experiments/eval.yaml --selector oracle|policy [--policy <ckpt>] [--split dev] [--tiers strict relaxed] [--limit N]`
— built in P4: same rollout/materialiser/reader for every selector, `metric_only`, both tiers
by default; writes `predictions_<tier>.jsonl`, `result.json` (suite + slices + reader
calls/hour) and the reader shard; `--reported` refuses the mock reader and the oracle selector.
`python -m mhtable.storage push <repo> <local> <path in repo>` (notebook uploads).

## 7. End-to-end flows

**Sourcing (once, molab):** download Spider (Google Drive), MMQA mirror (Google Drive), BIRD
(official zips) → checksum → push to `mhtable-raw`.

**Build (molab, `notebooks/build_data.py`; laptop only for `--limit` dev runs):** fetch raw →
loaders → materialise MMQA DBs → execute gold SQL + tier analysis → split → (P2) oracle labels →
(P3) edge and cell indexes → push `mhtable-data`, tag `data-vN`. A full rebuild of both families
(data + index from the local build folders + verification, tag on a separate button) is one
notebook, `notebooks/rebuild_v3.py`. The tag is created only after the index is pushed, so a
`data-vN` tag always covers data and index together.

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
- **BIRD's largest databases are slow without indexes** (`movie_platform`, `talkingdata`: tables
  of millions of rows). data-v4 adds SQLite indexes (`env/sqlindex.py`); without them look-aheads
  time out (−1) and the environment check ran 90+ min without finishing one tier.

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
