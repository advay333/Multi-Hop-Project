"""
The reader prompt, shared by every reader and every selector (PLAN.md §6 rule 6).
It is built from the question text and the serialised sub-table only — plain strings, so no
gold record can be passed in (firewall, DESIGN.md §8).

Versions (`reader.prompt`; part of the reader id, so cached answers never mix):
  p1  the original short instruction (P4 gate, 2026-11-19)
  p2  explicit rules aimed at the gate's miss types: exactly the requested attributes, every
      matching row for lists, compare all rows for superlatives / counts
"""

from __future__ import annotations

from mhtable.firewall import FirewallError

P1 = """You are given a table extracted from a database and a question about it.
Answer using only the table. Give just the answer value(s): one answer per line; if an answer
has several fields, separate them with " | ". Do not explain.

Table:
{subtable}

Question: {question}
Answer:"""

P2 = """You are given a table extracted from a database and a question about it. Answer using only the table.

Rules:
- Output only the answer. No explanation, no headers, no bullets.
- Write one answer per line. If the question asks for several attributes of each answer (for example a name and an age), put them on the same line separated by " | ", in the order the question asks for them.
- Give exactly the attributes the question asks for. Do not add other columns, such as the values used to filter or rank the rows.
- If the question asks for several items, go through every row of the table and list every row that satisfies the question. Do not stop early and do not repeat an item.
- For "how many" questions answer with the number only. For highest / lowest / most / least questions, compare all rows before answering.

Table:
{subtable}

Question: {question}
Answer:"""

TEMPLATES = {"p1": P1, "p2": P2}
TEMPLATE = P1  # backwards-compatible name


def build_prompt(question: str, subtable: str, version: str = "p1") -> str:
    if not isinstance(question, str) or not isinstance(subtable, str):
        raise FirewallError("the reader prompt is built from the question and sub-table strings only")
    if version not in TEMPLATES:
        raise ValueError(f"unknown prompt version {version!r}; known: {sorted(TEMPLATES)}")
    return TEMPLATES[version].format(subtable=subtable, question=question)
