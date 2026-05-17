import os
from pathlib import Path
from typing import Any, Mapping
import yaml


class ConfigDict(dict):
    """Nested dict with attribute access.

    cfg.train.lr == cfg["train"]["lr"]
    """
    def __init__(self, mapping: Mapping[str, Any] | None = None):
        super().__init__()
        if mapping is None:
            return
        for k, v in mapping.items():
            self[k] = ConfigDict(v) if isinstance(v, Mapping) else v

    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        raise AttributeError(f"ConfigDict has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = ConfigDict(value) if isinstance(value, Mapping) else value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike) -> ConfigDict:
    """Load a YAML file, recursively resolving `extends: <relative-path>`.

    `extends` may be a string or list of strings. Each base file is loaded
    first and then deep-merged with the current file's keys winning.
    """
    path = Path(path).resolve()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    parents = raw.pop("extends", None)
    if parents is None:
        merged = raw
    else:
        if isinstance(parents, str):
            parents = [parents]
        merged: dict = {}
        for parent in parents:
            parent_path = (path.parent / parent).resolve()
            merged = _deep_merge(merged, dict(load_config(parent_path)))
        merged = _deep_merge(merged, raw)

    return ConfigDict(merged)
