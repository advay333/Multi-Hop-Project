# Task 01 — Dataset Construction (Spider → training corpus, MMQA → eval)

**Prerequisite reading:** `PROJECT.md` §1 (problem), §3 (MDP), §8 (datasets, gates,
leakage). This document is self-contained for *implementation* but the design rationale
lives there.

**Owner:** Environment/eval lead (person B). **Weeks:** 1–3. **Runtime:** 2–4 h compute
once the code works.

**Deliverable:** `data/` fully populated, all gates in §9 reported, and a written
`data/DATASET_CARD.md`.

**Hard requirement: zero human annotation.** Every label in this pipeline comes from
either (a) Spider's existing 2018 human annotations, or (b) executing SQL against
SQLite. If you find yourself needing a human to label something, the design is wrong —
re-read §5.

---

## 1. What we need and why

Three artefacts:

| Artefact | Purpose | Size target |
|---|---|---|
| `data/raw/train_synth.json` | **training corpus** for BC and PPO | 20k–50k instances |
| `data/raw/mmqa.json` (mirrored) | **eval set**, held out entirely | 3,312 instances |
| `data/trajectories/oracle_traj.jsonl` | per-hop gold entity sets | one row per (instance, hop) |

**Why we generate a training corpus at all.** MMQA has 3,312 instances. On-policy PPO at
H=2–4 needs 50k–200k episodes. 3,312 is an *eval set*, not a training set. For
calibration: StepSearch achieved its gains with 19k training examples; GlobalRAG with
8k. 20k puts us in the normal range.

**Why this is a deliverable, not a workaround.** A Spider-derived multi-table corpus with
gold per-hop entity sets and paraphrase-hardened question variants is reusable by anyone
working on this task, independent of whether our PPO beats our baselines. Treat it
accordingly: version it, document it, release it.

---

## 2. Source data

### 2.1 Spider (`taoyds/spider`, EMNLP 2018)

```
spider/database/<db_id>/<db_id>.sqlite   real SQLite files with real rows
spider/tables.json                       schemas + primary_keys + foreign_keys
spider/train_spider.json                 NL/SQL pairs (not directly needed)
spider/dev.json                          NL/SQL pairs (reserve for proxy eval)
```

200+ databases, 138 domains, 5,693 SQL queries.

**`tables.json` is the important file.** It ships `primary_keys` (list of column
indices) and `foreign_keys` (list of `[col_idx, col_idx]` pairs), human-labelled by the
Spider authors. **This is why we need no annotation** — MMQA paid two CS experts to
annotate PKs and FKs, but Spider already had them.

### 2.2 MMQA (ICLR 2025)

- Paper: `proceedings.iclr.cc/paper_files/paper/2025/file/794a425a2e47e05d29d30f79b79a692d-Paper-Conference.pdf`
- Data: `https://anonymous.4open.science/r/MMQA-34B1`

Per instance: NL question, gold SQL, gold answer, FK annotations, PK annotations.
Subsets: 2591 2-table + 721 3-table instances over ~710 unique tables.

**GATE G-1, DAY ONE, BEFORE ANYTHING ELSE.** That is anonymised-review hosting, not a
permanent archive. Those links expire.

```bash
# do this first
mkdir -p data/raw/mmqa_mirror
# download every file, commit the mirror, record the date and a checksum
```

If dead: email Jian Wu (Institute of Science Tokyo). If that fails, §8 describes
building a proxy eval set from Spider *dev* databases. The generator in §5 is exactly
what makes that fallback viable.

**Name collision warning.** Three unrelated datasets are called "MMQA": this one,
AllenAI's MultiModalQA (`allenai/multimodalqa`), and a 2018 multilingual QA set
(`deepaknlp/MMQA`). Search results are polluted. Always disambiguate by "multi-table
multi-hop, ICLR 2025".

### 2.3 MultiTabQA (Pal et al., ACL 2023) — do not rewrite the templates

MMQA's construction: *"we follow the paradigm of Pal et al. (2023) to synthesize
multi-table SQL queries through the 45 manually crafted templates over the Spider
database and hand-crafted rules for operation types."*

Those templates come from MultiTabQA, which has:
- repo: `github.com/kolk/MultiTabQA`
- a released pre-training corpus of **132,645 SQL queries with tabular answers**
- HuggingFace: `vaishali/spider-tableQA-pretraining`

**Start from these.** Write templates only for join shapes they do not cover (gate G-4).
This can save a week.

---

## 3. Output schema

Fix this before writing any generator code. Both `train_synth.json` and the normalised
MMQA file use it.

```json
{
  "instance_id": "synth_dept_mgmt_0041",
  "source": "spider_synth",
  "db_id": "department_management",
  "n_hops": 3,
  "question": "What are the distinct creation years of the departments managed by a secretary born in state Alabama?",
  "question_hardened": "What are the different years the departments run by a secretary from Alabama were set up?",
  "gold_sql": "SELECT DISTINCT T3.Creation FROM Head AS T1 JOIN Management AS T2 ON T1.Head_ID = T2.Head_ID JOIN Department AS T3 ON T2.Department_ID = T3.Department_ID WHERE T1.Born_State = 'Alabama'",
  "gold_answer": ["1903"],
  "question_category": "Select",
  "join_path": [
    {"from_table": "Head",       "from_col": "Head_ID",
     "to_table":   "Management", "to_col":   "Head_ID"},
    {"from_table": "Management",  "from_col": "Department_ID",
     "to_table":   "Department",  "to_col":   "Department_ID"}
  ],
  "start_table": "Head",
  "predicate": {"table": "Head", "column": "Born_State",
                "op": "=", "literal": "Alabama"},
  "gold_entity_sets": [
    {"hop": 1, "column": "Head.Head_ID",            "values": ["1"]},
    {"hop": 2, "column": "Management.Department_ID", "values": ["7"]},
    {"hop": 3, "column": "Department.Creation",      "values": ["1903"]}
  ],
  "pk": {"Head": ["Head_ID"], "Department": ["Department_ID"]},
  "fk": [["Management.Head_ID", "Head.Head_ID"],
         ["Management.Department_ID", "Department.Department_ID"]],
  "provenance": {
    "template_id": "join3_filter_distinct",
    "generator_model": "Qwen2.5-7B-Instruct",
    "roundtrip_verified": true,
    "roundtrip_verifier": "Qwen2.5-7B-Instruct",
    "hardened_verified": true
  }
}
```

Notes:

- `question_category` ∈ {Select, Numerical, List, Count} — match MMQA's distribution
  (2-table: 1289 Select / 889 Numerical / 214 List / 200 Count; 3-table: 346 / 289 / 44
  / 42).
- `question_hardened` is the paraphrase with schema vocabulary removed (§7). Nullable
  but strongly preferred.
- `gold_entity_sets` values are strings. Normalise consistently (see §6.3).
- `provenance` exists so a reviewer can filter the corpus by how it was verified.
  Do not omit it.

---

## 4. Pipeline overview

```
Stage A  ingest Spider + mirrored MMQA           -> data/db/, schema graphs
Stage B  enumerate join paths (FK graph walk)    -> candidate paths per DB
Stage C  instantiate SQL from templates          -> executable gold SQL
Stage D  SQL -> natural language + verify        -> question, question_hardened
Stage E  execute for answers + oracle labels     -> gold_answer, gold_entity_sets
Stage F  split by database + embedding cache     -> data/splits/, data/cache/
```

Stages A, B, C, E, F are fully deterministic and human-free. Stage D uses a generative
model but is **verified by execution**, not by a human.

---

## 5. Stage-by-stage implementation

### Stage A — Ingest

```python
# src/schema_graph.py
class SchemaGraph:
    """Built from spider/tables.json. Nodes = tables, edges = FK pairs."""
    def __init__(self, db_entry: dict): ...
    def edges_from(self, table: str) -> list[FKEdge]: ...
    def is_pk(self, table: str, col: str) -> bool: ...
    def simple_paths(self, length: int) -> Iterator[list[FKEdge]]: ...
```

`tables.json` gives `foreign_keys` as index pairs into a flat `column_names` list.
Resolve those indices to `(table_name, column_name)` once and cache.

**Self-joins.** If the gold SQL aliases the same table as T1 and T2, that is ONE node in
the graph. Decide now whether to support self-join paths (recommendation: exclude them
from generation, record them in MMQA eval as a known coverage gap).

**Composite FKs.** MMQA averages 2.81 FKs per table. A composite key is a *set* of column
pairs forming one edge. Model an `FKEdge` as a list of column pairs, not a single pair,
even if you only generate single-column joins initially.

Persist to `data/cache/schema_graphs.pkl`. Copy or symlink SQLite files into `data/db/`.

**Report G-2 here** (see §9): median AND mean rows per table, per DB and overall.

### Stage B — Enumerate join paths

```python
for db in databases:
    G = schema_graph(db)
    for path in G.simple_paths(length=2) + G.simple_paths(length=3):
        if all(edge.is_fk for edge in path):
            n = execute(f"SELECT COUNT(*) FROM {render_join(path)}")
            if n > 0:
                yield (db, path, n)
```

Cheap and exhaustive. No templates needed yet. Record the join cardinality `n` — it is
both a filter and a useful feature later.

**Report G-3:** number of viable paths per DB and in total. If yield is low, add BIRD
databases (95 DBs, larger and messier) before proceeding.

### Stage C — Instantiate SQL

For each `(path, template)` pair:

1. Choose a projection column (the answer column) — typically on the last table.
2. Choose a filter column on the **start** table.
3. **Sample the literal from actual cell values in that column.** This is what makes the
   question answerable at all.

```sql
SELECT {agg}({proj}) FROM {t1}
  JOIN {t2} ON {fk1}
  [JOIN {t3} ON {fk2}]
 WHERE {filter_col} {op} '{sampled_real_value}'
```

**Reject-sample on execution.** Discard if the result set is empty or larger than ~50
rows. Empty means unanswerable; huge means the question is degenerate.

Vary `{agg}` and `{op}` to hit all four question categories:

| Category | Shape |
|---|---|
| Select | `SELECT DISTINCT col ... WHERE col2 = 'lit'` |
| List | `SELECT col_a, col_b ... ORDER BY / LIMIT` |
| Count | `SELECT COUNT(*) ...` |
| Numerical | `SELECT AVG/SUM/MAX(col) ...` or `WHERE numeric_col > lit` |

**Include non-equality predicates.** MMQA has questions like "stations with latitude
above 37.4". If the generator only emits equality, the action space never learns range
filters and MMQA eval coverage suffers.

### Stage D — SQL → natural language, verified by execution

The only generative step. Use a **local** model (Qwen2.5-7B-Instruct) — this runs tens of
thousands of times and an API bill is unnecessary.

Prompt with: schema (table + column names), the SQL, 3–5 few-shot examples drawn from
real MMQA questions. Ask for JSON: `{"question": "..."}`.

Then verify **automatically, three ways**:

1. **Round-trip check (the important one).** Ask a model — fresh context, ideally a
   different model — to write SQL from the generated question given only the schema.
   Execute both SQL queries. Keep the instance only if the result sets match
   (set-equality after normalisation). This is execution-verified filtering, the same
   trick Arctic-Text2SQL-R1 uses for its reward. **No human in the loop.**
2. **Literal presence.** The sampled value ("Alabama") must appear in the question
   (allow case/format variation). Otherwise the question is unanswerable.
3. **Deduplication.** Normalise whitespace and case; drop duplicate question texts.

**Expect to keep 50–70%. Generate 3–4× the target and let the filter work.**
**Report G-5:** the round-trip pass rate. Below 40% → improve the prompt or swap the
generator model before scaling up.

Then generate `question_hardened` — see §7. It goes through the same three checks.

### Stage E — Answers and oracle labels, all by execution

```python
gold_answer      = execute(gold_sql)
gold_entity_sets = [execute(prefix_t) for t in 1..H]
join_path        = known by construction (Stage B)
pk, fk           = from spider/tables.json
```

**Prefix construction.** Truncate the join chain at hop `t` and re-project onto the
column that hop `t+1` will join on:

```sql
-- hop 1: project the column hop 2 joins on
SELECT DISTINCT T1.Head_ID FROM Head AS T1 WHERE T1.Born_State='Alabama';
-- hop 2
SELECT DISTINCT T2.Department_ID FROM Head T1
  JOIN Management T2 ON T1.Head_ID=T2.Head_ID WHERE T1.Born_State='Alabama';
-- hop 3 (terminal): project the answer column
SELECT DISTINCT T3.Creation FROM ... JOIN Department T3 ON ...;
```

**CRITICAL — strip the SELECT clause for non-terminal hops.** Define `E_t^gold` from the
**FROM / JOIN / WHERE** structure only. The selector's job is to fetch the right *rows*;
the reader does the arithmetic. Without this, every aggregation question
(`SELECT AVG(salary)`) becomes unrepresentable, because no candidate action produces a
result set equal to `{37.4}`. MMQA has 889+289 Numerical and 200+42 Count questions —
this single decision is the difference between ~1,900 and ~3,300 usable eval instances.
See `PROJECT.md` §8.5.

**For synthetic instances the join path is known by construction**, so no parsing is
needed. `sqlglot` parsing is only required for **MMQA eval instances**, where the SQL
arrived pre-written:

```python
import sqlglot
ast  = sqlglot.parse_one(gold_sql, read="sqlite")
joins = ast.find_all(sqlglot.exp.Join)      # ON clauses -> edges
where = ast.find(sqlglot.exp.Where)         # predicate literals
```

Emit `data/trajectories/oracle_traj.jsonl`, one row per (instance, hop), including the
matched candidate-action index once `env.candidate_actions` exists (Week 2). That
matching step is where **G-6 coverage** gets measured: the oracle says which entity set
you should hold; you must find which enumerated action produces it. Sometimes none does
exactly — log those and report the rate.

### Stage F — Split and cache

**Split by DATABASE, never by question.**

```
Spider train DBs (~140)  -> data/splits/train_dbs.txt
Spider dev DBs           -> held in reserve (proxy eval fallback)
All MMQA instances       -> data/splits/dev_dbs.txt, test_dbs.txt
```

Splitting by question puts the same schema on both sides and makes every generalisation
number meaningless.

Then the embedding cache — the only GPU work in this task:

```python
# for every column header and every sampled cell value, in ONE batched pass
texts = [f"{tbl}.{col}" for ...] + [str(v) for v in sampled_values]
embs  = encoder.encode(texts, batch_size=512)   # bge-m3 or gte-large
np.savez("data/cache/embeddings.npz", keys=texts, vecs=embs)
```

After this, `featurise.py` is a dictionary lookup forever. **This is what makes the whole
project fit on one GPU.** Sample a bounded number of distinct values per column (e.g.
200) rather than embedding every cell.

---

## 6. Rules that will otherwise bite you

### 6.1 No database-specific integer identifiers, anywhere

Not in the JSON, not in the cache keys, not in the featuriser. Every reference is by
`table_name.column_name` string. A learned per-table or per-column ID makes Week 10
transfer chance-level with no diagnosable cause. See `PROJECT.md` §4.2.

### 6.2 Gold SQL is a training-time-only signal

`gold_sql` and `gold_entity_sets` exist in the JSON, so it is *physically possible* for
them to leak into a reader prompt or a state feature. Enforce the firewall in code:

```python
# featurise.py
FORBIDDEN = {"gold_sql", "gold_entity_sets", "gold_answer", "join_path", "predicate"}
assert not (FORBIDDEN & set(state_dict.keys()))
```

### 6.3 Normalise values consistently, once

Entity-set comparison (`Jaccard(E_t, E_t^gold)`) is only meaningful if both sides are
normalised identically. Write ONE function and use it everywhere:

```python
def norm_value(v) -> str:
    # strip, casefold, collapse whitespace
    # numerics: canonical decimal, no trailing zeros ("7" == "7.0" == " 7 ")
    # NULL -> "<null>"
    # dates -> ISO-8601
```

`"7"` vs `"7.0"` vs `" 7 "` failing to match will silently zero out your shaping reward
and you will spend days looking for a bug in PPO.

### 6.4 Reproducibility

Seed everything. Record generator model name and version in `provenance`. Write
`data/DATASET_CARD.md` with: source versions, template list, per-stage yield counts, all
gate numbers, the generator and verifier models, and the split definition.

---

## 7. Schema-vocabulary leakage — handle this in the generator

**Read `PROJECT.md` §8.6 before implementing this.** Short version:

MMQA's questions were generated *from* SQL, so they leak column names. The authors
measured it: manually paraphrasing 100 questions to remove schema hints dropped
o1-preview from 43.5 → 40.7 EM (2-table) and 39.8 → 34.4 (3-table). Their example: the
original says "heads whose temporary acting value is 'Yes'" — which hands you the column
`temporary_acting`. The paraphrase says "department heads who are acting now" — which
does not.

**This hits our policy harder than it hits an LLM.** The action scorer is a dot product
between question embedding and column-name embedding. If the question literally contains
the column name, that dot product is doing string matching, not reasoning.

**What the generator must do:**

1. Sample the filter literal **after** choosing the column, so the *value* is in the
   question but the *column name* need not be.
2. Produce `question_hardened`: a second variant, instructed to avoid schema vocabulary
   (no column names, no table names), verified by the same round-trip check.
3. Emit a `leakage_score` per instance — e.g. token overlap between the question and the
   set of column names on the join path. Cheap, and it lets the RL lead stratify results
   by leakage without regenerating anything.

**Report G-7:** the EM gap between original and hardened variants, once baselines exist
(Week 3–4).

The companion diagnostic (header ablation → `col_0, col_1, ...`) is a featuriser flag,
not a dataset concern — but the dataset must not make it impossible, which means keeping
`question_hardened` around.

---

## 8. If MMQA is unavailable

In priority order:

1. **Mirror on day one** so this never arises.
2. Email the authors (Jian Wu, Institute of Science Tokyo; Linyi Yang, UCL).
3. **Build a proxy eval set from Spider `dev` databases** using this exact pipeline. Dev
   DBs are disjoint from train DBs, so the split remains clean. Cost: MMQA's published
   baseline numbers are no longer directly comparable — say so plainly in the paper and
   report our own baselines instead. The pipeline makes this a one-day pivot rather than
   a project-ending event.
4. **Fall back to BIRD** as the eval target: 12,751 question-SQL pairs across 95 large
   real databases with dirty data and an active leaderboard. Harder and better
   maintained. Downside: no multi-hop question framing and no MMQA baselines, so the
   whole eval protocol must be built from scratch. **Plan B, not Plan A.**

---

## 9. Gates — report every number in `DATASET_CARD.md`

| id | Gate | How to measure | Threshold / fail response |
|---|---|---|---|
| G-1 | MMQA availability | fetch the 4open.science link; checksum the mirror | dead → §8 |
| G-2 | Row counts | median AND mean rows per table, per DB and overall | if median is small while mean is large, the "context length" motivation rests on outlier DBs — reframe around join-space size and say so |
| G-3 | Path yield | count of viable 2- and 3-hop FK paths per DB | too low → add BIRD databases |
| G-4 | Template coverage | do MultiTabQA's released queries cover our join shapes? | write templates only for gaps |
| G-5 | Round-trip pass rate | % of Stage D output surviving verification | <40% → improve prompt or swap generator |
| G-6 | Oracle coverage | % of hops where some candidate action reaches `E_t^gold` | <85% → widen action space (add `carry_all`, range predicates) |
| G-7 | Leakage gap | EM difference, original vs hardened, on a baseline | large gap → header-ablation becomes a headline experiment |
| — | Category balance | distribution over Select/Numerical/List/Count | should approximate MMQA's |
| — | Corpus size | instances after all filtering | target ≥20k; hard floor 8k |

**G-2 and G-6 are the two that change the project if they fail.** G-2 undermines the
motivation; G-6 caps both BC and PPO. Measure both before Week 5.

---

## 10. Definition of done

- [ ] MMQA mirrored, checksummed, date recorded
- [ ] `data/db/` populated; `data/cache/schema_graphs.pkl` built
- [ ] `data/raw/train_synth.json` ≥20k instances, schema in §3, all with `provenance`
- [ ] `question_hardened` present for ≥80% of instances
- [ ] `data/raw/mmqa.json` normalised into the same schema
- [ ] `data/trajectories/oracle_traj.jsonl` emitted with per-hop gold entity sets
- [ ] `data/cache/embeddings.npz` built; `featurise` confirmed as pure dict lookup
- [ ] `data/splits/*.txt` — split by DATABASE, verified disjoint
- [ ] `data/DATASET_CARD.md` with every gate number from §9
- [ ] Firewall assertions in place (§6.2)
- [ ] `norm_value` written once and used everywhere (§6.3)
- [ ] Regeneration is one command with a fixed seed


