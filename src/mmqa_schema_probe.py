"""
One-off inspection tool for the MMQA mirror files (Phase 0, step 0c).

The original MMQA data link in the paper (anonymous.4open.science/r/MMQA-34B1) is dead.
The replacement is a Google Drive folder with two large JSON files
(Synthesized_two_table.json, Synthesized_three_table.json) whose exact internal schema
is unknown. Do NOT json.load() these files whole — they are hundreds of MB and can
exhaust memory on a laptop. This script streams with ijson instead.

Usage:
    python src/mmqa_schema_probe.py data/raw/mmqa_mirror/Synthesized_two_table.json
    python src/mmqa_schema_probe.py data/raw/mmqa_mirror/Synthesized_three_table.json
"""

import ijson
import json
import sys
from pathlib import Path


def peek_top_level_shape(path: Path) -> str:
    """Read a few raw bytes to determine if the file is a JSON array or object."""
    with open(path, "rb") as f:
        chunk = f.read(4096).lstrip()
    if chunk.startswith(b"["):
        return "array"
    if chunk.startswith(b"{"):
        return "object"
    return "unknown"


def summarize_value(v, max_len=200):
    """Truncate long values (e.g. embedded table rows) for readable printing."""
    s = json.dumps(v, ensure_ascii=False, default=str)
    if len(s) > max_len:
        return s[:max_len] + f"... <truncated, full length {len(s)}>"
    return s


def describe_record(record: dict, indent=2):
    pad = " " * indent
    print(f"{pad}Top-level keys: {sorted(record.keys())}")
    for k, v in record.items():
        if isinstance(v, dict):
            print(f"{pad}  {k}: dict with keys {sorted(v.keys())}")
        elif isinstance(v, list):
            n = len(v)
            elem_desc = ""
            if n > 0:
                elem = v[0]
                if isinstance(elem, dict):
                    elem_desc = f", element[0] keys={sorted(elem.keys())}"
                else:
                    elem_desc = f", element[0]={summarize_value(elem, 80)}"
            print(f"{pad}  {k}: list, len={n}{elem_desc}")
        else:
            print(f"{pad}  {k}: {type(v).__name__} = {summarize_value(v, 120)}")


def probe(path: Path, n_records: int = 5):
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"=== {path.name} ({size_mb:.1f} MB) ===")

    shape = peek_top_level_shape(path)
    print(f"Top-level JSON shape: {shape}\n")

    with open(path, "rb") as f:
        if shape == "array":
            items = ijson.items(f, "item")
        elif shape == "object":
            # Could be {db_id: {...}} or {"data": [...]} etc. Stream key/value pairs
            # at depth 1 and treat each value as a "record" for inspection purposes.
            items = ((k, v) for k, v in ijson.kvitems(f, ""))
        else:
            print("Could not determine top-level shape from first 4KB — inspect manually.")
            return

        count = 0
        for entry in items:
            if shape == "object":
                key, record = entry
                print(f"[entry {count}] key={key!r}")
                if isinstance(record, dict):
                    describe_record(record)
                else:
                    print(f"  value: {type(record).__name__} = {summarize_value(record, 200)}")
            else:
                record = entry
                print(f"[record {count}]")
                if isinstance(record, dict):
                    describe_record(record)
                else:
                    print(f"  {type(record).__name__} = {summarize_value(record, 200)}")
            print()
            count += 1
            if count >= n_records:
                break

    if count == 0:
        print("No records found — check the shape-detection logic against this file.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    probe(Path(sys.argv[1]))
