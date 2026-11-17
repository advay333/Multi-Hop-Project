"""
The one reader prompt, shared by every reader and every selector (PLAN.md §6 rule 6).
It is built from the question text and the serialised sub-table only — plain strings, so no
gold record can be passed in (firewall, DESIGN.md §8).
"""

from __future__ import annotations

from mhtable.firewall import FirewallError

TEMPLATE = """You are given a table extracted from a database and a question about it.
Answer using only the table. Give just the answer value(s): one answer per line; if an answer
has several fields, separate them with " | ". Do not explain.

Table:
{subtable}

Question: {question}
Answer:"""


def build_prompt(question: str, subtable: str) -> str:
    if not isinstance(question, str) or not isinstance(subtable, str):
        raise FirewallError("the reader prompt is built from the question and sub-table strings only")
    return TEMPLATE.format(subtable=subtable, question=question)
