"""
Parses the MMQA mirror files (data/raw/mmqa_mirror/Synthesized_{two,three}_table.json)
into the unified instance schema defined in TASK_01_DATASET.md §3.

SCHEMA FINDINGS (from src/mmqa_schema_probe.py, run against both mirror files on
2026-08-12 — see data/DATASET_CARD.md for the checksum/date this was verified against):

Each file is a top-level JSON array. One element looks like:

    {
      "id_": 0,
      "Question": "What are the names of heads serving as temporary acting heads "
                   "in departments with rankings better than 5?",
      "SQL": "SELECT head.head_Name FROM head INNER JOIN management ON "
             "head.head_ID = management.head_ID INNER JOIN department ON "
             "management.Department_ID = department.Department_ID WHERE "
             "department.Ranking < 5 AND management.temporary_acting = 'Yes';",
      "answer": "Jeff Maggert, Billy Mayfair",
      "table_names": ["department", "head", "management"],
      "tables": [
        {"table_columns": ["Department_ID", "Department_Name", "Creation", ...],
         "table_content": [[1, "State", "1789", 1, "9.96", "30266.0"], ...]},
        ...  # one entry per table_names, same order
      ],
      "primary_keys": ["Department_ID", "head_ID", "Department_ID"],   # one per table, same order as table_names
      "foreign_keys": ["head_ID", "Department_ID"]                     # one per join edge (n_tables - 1), NOT (table,col) pairs
    }

This differs from what TASK_01_DATASET.md §2.2 assumed (a small annotation file
referencing separate Spider .sqlite files). Consequences for this ingester:

1. **No db_id field.** MMQA instances don't name their source Spider database. Since
   full row content is embedded (see point 2), we don't need the original .sqlite files
   to execute anything — but Gate rule "split by DATABASE, not by question" (§5 Stage F)
   still requires grouping instances that share an underlying schema. Fix: derive a
   `schema_signature` = hash of (sorted table_names, sorted table_columns per table).
   Instances with the same signature are treated as the same "database" for split
   purposes, even without recovering the real Spider db_id.

2. **Full table content is embedded per instance**, not referenced. `table_content` is a
   list of rows, each row a positional list matching `table_columns` (NOT a list of
   dicts). This is why Synthesized_two_table.json is 838 MB despite covering only 2591
   instances — every instance carries its own copy of its tables' rows. Stage A for
   MMQA is therefore "materialize a fresh in-memory/temp SQLite DB per instance from
   table_columns + table_content," not "point at a pre-existing .sqlite file."

3. **Row values are untyped / inconsistently stringified.** Observed: ints (`1`),
   plain strings (`"State"`), and numeric-looking strings with float suffixes
   (`"9.96"`, `"30266.0"`, `"67.0"`) all in the same row. Do not assume a column's
   declared type from `table_columns` — infer per-value at insert time, and lean on
   `norm.py`'s numeric canonicalization (`"7"` == `"7.0"`) rather than SQLite type
   affinity to make cross-value comparisons correct.

4. **`answer` is a single string, not a list** (e.g. `"Jeff Maggert, Billy Mayfair"` for
   a multi-row List question) — comma-joined, but splitting on `", "` is unsafe if a
   value itself contains a comma. Do NOT parse this field for `gold_answer`. Instead,
   materialize the SQLite DB (point 2) and execute `SQL` directly, exactly as Stage E
   already does for everything else — this also gives properly typed values. Keep the
   original `answer` string only as an audit field: log the % of instances where our
   fresh execution's flattened result matches the provided string, as a sanity check on
   the materialization step (deviations point at a table-reconstruction bug, not a
   labeling error).

5. **`foreign_keys` / `primary_keys` are flat, bare column-name lists — NOT positionally
   parallel to `table_names`, and NOT reliably one-per-table/one-per-edge.** Checked
   against Synthesized_three_table.json record 0 (3 tables): `primary_keys` had 3
   entries and looked one-per-table — but Synthesized_two_table.json (2 tables) has
   records with `primary_keys` of length 3 and 4, and `foreign_keys` of length 2 and 3.
   The only safe reading: these are just "all PK column names that appear somewhere in
   this instance's tables" and "all FK column names that appear somewhere," with
   composite keys contributing multiple entries for a single table/edge. Do NOT assume
   `primary_keys[i]` <-> `table_names[i]`.
   Resolve purely by name-matching: for each FK/PK column name, find which table(s) in
   `tables[*].table_columns` contain a same-named column and treat those as the
   candidate table(s) it belongs to. Log any name that resolves to 0 or >2 tables (for
   FKs) as a coverage-gap case rather than guessing.
   **Also normalize before matching**: casing and separators are inconsistent *between
   the two files* — `Synthesized_three_table.json` uses source casing with underscores
   (`"head_ID"`, `"Department_ID"`), `Synthesized_two_table.json` uses lowercased words
   with spaces (`"head id"`, `"department id"`). Write one normalizer
   (lowercase + collapse `[ _]+` to a single form) and match both column names and
   FK/PK strings through it — do not compare them raw.

6. **SQL uses real table names, not aliases** (`head.head_Name`, not `T1.col`) — this
   actually simplifies `sqlglot` parsing versus the aliased form PROJECT.md's example
   assumed; no alias-resolution step needed for MMQA's join_path/predicate extraction.

TODO (implementation, not yet written):
- build_sqlite_from_instance(record) -> sqlite3.Connection (in-memory), using finding 2/3
- resolve_join_graph(table_names, tables, foreign_keys) -> list[FKEdge], using finding 5
- normalize_instance(record) -> dict matching TASK_01_DATASET.md §3 schema, using
  Stage E's oracle extraction (already built in Phase 1) for gold_answer/gold_entity_sets
  instead of the raw `answer` string
- schema_signature(table_names, tables) -> str, for Stage F's DB-level split
"""
