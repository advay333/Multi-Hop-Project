"""
Source loaders: each turns one raw source into a `SourceBundle` the build can process the same
way (execute gold SQL, tag tiers, split, write).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mhtable.data.schema import SchemaGraph


@dataclass
class RawExample:
    source: str  # spider / bird / mmqa
    ext_id: str  # position in the source files, e.g. "train_spider42", "2t17"
    db_id: str  # source database, the split grouping key
    db_ref: str  # storage key of the SQLite file
    question: str
    sql: str
    evidence: str | None = None
    extra: dict = field(default_factory=dict)  # source-specific audit fields

    @property
    def instance_id(self) -> str:
        return f"{self.source}:{self.db_id}:{self.ext_id}"


@dataclass
class SourceBundle:
    examples: list[RawExample] = field(default_factory=list)
    schemas: dict[str, SchemaGraph] = field(default_factory=dict)  # db_ref -> graph
    db_files: dict[str, Path] = field(default_factory=dict)  # db_ref -> local SQLite path
    spider_candidates: dict[str, list[str]] = field(default_factory=dict)  # MMQA db_ref -> Spider db_ids
    stats: dict = field(default_factory=dict)
