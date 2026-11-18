"""
Pieces shared by the real LLM readers (`hf`, `vllm`; report E.1: Qwen2.5-7B-Instruct, greedy).

    reader_id(...)     the cache identity: backend, model, precision, decoding, prompt version.
                       Anything that can change an answer is in it, so cache shards from different
                       setups never mix (PLAN.md §4b fairness rule).
    chat_prompt(...)   the shared prompt (readers/prompt.py) wrapped in the model's chat template;
                       a sub-table too long for the context window is cut row by row (counted).
    clean_answer(...)  strips the decoration models add around the requested answer format.
    ReaderStats        prompts, cut prompts, answers cut by max_new_tokens, seconds.

Real readers see only the question and sub-table strings; `meta` is never used.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from mhtable.readers.prompt import build_prompt

PROMPT_VERSION = "p1"  # bump whenever readers/prompt.py TEMPLATE or chat_prompt changes


def reader_id(backend: str, model: str, dtype: str, max_new_tokens: int, quantization: str | None = None) -> str:
    q = f":{quantization}" if quantization else ""
    return f"{backend}:{model}:{dtype}{q}:greedy:max{max_new_tokens}:{PROMPT_VERSION}"


@dataclass
class ReaderStats:
    prompts: int = 0
    prompts_cut: int = 0  # sub-table rows dropped to fit the context window
    answers_cut: int = 0  # generation stopped by max_new_tokens
    prompt_tokens: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["calls_per_hour"] = round(3600 * self.prompts / self.seconds, 1) if self.seconds > 0 else None
        d["answers_cut_fraction"] = round(self.answers_cut / self.prompts, 4) if self.prompts else 0.0
        return d


def _render(tok, text: str) -> str:
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return text  # models without a chat template (tiny test models)


def chat_prompt(tok, question: str, subtable: str, max_prompt_tokens: int) -> tuple[str, int, bool]:
    """-> (prompt text, prompt tokens, whether sub-table rows were dropped to fit)."""
    text = _render(tok, build_prompt(question, subtable))
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    if n <= max_prompt_tokens:
        return text, n, False
    lines = subtable.split("\n")
    lo, hi = 2, len(lines)  # keep the header and separator; binary search the rows that fit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        t = _render(tok, build_prompt(question, "\n".join(lines[:mid])))
        if len(tok(t, add_special_tokens=False)["input_ids"]) <= max_prompt_tokens:
            lo = mid
        else:
            hi = mid - 1
    text = _render(tok, build_prompt(question, "\n".join(lines[:lo])))
    return text, len(tok(text, add_special_tokens=False)["input_ids"]), True


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_PREFIX = re.compile(r"^\s*(final\s+)?answer\s*:\s*", re.IGNORECASE)


def clean_answer(text: str) -> str:
    """Drop code fences, a leading "Answer:", list bullets and blank lines; keep one answer per line."""
    t = _FENCE.sub("", (text or "").strip())
    t = _PREFIX.sub("", t)
    out = []
    for line in t.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        if line:
            out.append(line)
    return "\n".join(out)
