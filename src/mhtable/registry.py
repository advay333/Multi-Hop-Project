"""
Name -> class registries for every swappable component.

    @register("reader", "mock")
    class MockReader(Reader): ...

    reader = build("reader", cfg.reader)   # cfg.reader = {name: mock, ...other kwargs}

`build` passes the remaining keys of the config section as keyword arguments, plus any extra
objects the caller supplies (e.g. the env). Adding a component never requires editing the
training loop — only a new class and a config entry.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

KINDS = ("reader", "encoder", "policy", "algo", "selector", "dataset")

_REGISTRY: dict[str, dict[str, type]] = {k: {} for k in KINDS}


def register(kind: str, name: str) -> Callable[[type], type]:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown component kind {kind!r}; expected one of {KINDS}")

    def deco(cls: type) -> type:
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing is not cls:
            raise KeyError(f"{kind} {name!r} already registered to {existing.__qualname__}")
        _REGISTRY[kind][name] = cls
        return cls

    return deco


def get(kind: str, name: str) -> type:
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        known = sorted(_REGISTRY.get(kind, {}))
        raise KeyError(f"no {kind} named {name!r}; registered: {known}") from None


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY[kind])


def build(kind: str, section: Mapping[str, Any] | str, **extra: Any) -> Any:
    """Instantiate `section.name` with the section's other keys as kwargs."""
    if isinstance(section, str):
        section = {"name": section}
    params = dict(section)
    name = params.pop("name", None)
    if name is None:
        raise KeyError(f"{kind} config section needs a 'name' key, got keys {sorted(params)}")
    params = {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in params.items()}
    return get(kind, name)(**params, **extra)
