"""
Experiment configuration: YAML files with inheritance and dotted CLI overrides.

A config file may list parent files under `_base_`; parents are merged first (left to right),
then the file itself, then CLI overrides like `algo.clip=0.1 reader.name=mock`. The result is
an immutable `Config` that supports attribute access and is saved verbatim into every run, so
a run is reproducible from its `config.yaml` alone.

Components validate their own section when they are built (see registry.build), so this module
stays schema-agnostic.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class Config(Mapping):
    """Read-only nested mapping with attribute access (`cfg.algo.clip`)."""

    def __init__(self, data: Mapping[str, Any] | None = None):
        object.__setattr__(self, "_data", {k: _wrap(v) for k, v in (data or {}).items()})

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"config has no key {key!r} (keys: {sorted(self._data)})") from None

    def __setattr__(self, key, value):
        raise TypeError("Config is immutable; use with_overrides()")

    def __repr__(self):
        return f"Config({self.to_dict()!r})"

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Config) else _unwrap(v) for k, v in self._data.items()}

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def with_overrides(self, overrides: Iterable[str] | Mapping[str, Any]) -> "Config":
        data = self.to_dict()
        items = overrides.items() if isinstance(overrides, Mapping) else map(_parse_override, overrides)
        for key, value in items:
            _set_path(data, key, value)
        return Config(data)


def _wrap(v):
    if isinstance(v, Mapping) and not isinstance(v, Config):
        return Config(v)
    if isinstance(v, list):
        return tuple(_wrap(x) for x in v)
    return v


def _unwrap(v):
    if isinstance(v, Config):
        return v.to_dict()
    if isinstance(v, tuple):
        return [_unwrap(x) for x in v]
    return v


def _parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"override must look like key.path=value, got {item!r}")
    key, raw = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw)  # "0.1" -> 0.1, "true" -> True, "[1,2]" -> [1, 2]


def _set_path(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {dotted!r}: {part!r} is not a mapping")
    node[parts[-1]] = value


def deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_tree(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular _base_ include: {' -> '.join(map(str, seen + (path,)))}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = deep_merge(merged, _load_tree((path.parent / b), seen + (path,)))
    return deep_merge(merged, raw)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    return Config(_load_tree(Path(path))).with_overrides(list(overrides))
