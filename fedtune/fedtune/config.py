"""Configuration: defaults, YAML loading, command-line overrides, validation.

Every key has a type and an allowed range. A bad value is rejected with a
message that names the key and the allowed range (SRS UC1, FR-1, FR-2).
"""
from __future__ import annotations

import copy
from typing import Any

import yaml

STRATEGIES = ("fedavg", "diversity", "clustered", "hybrid", "fedsa")
PARTITION_MODES = ("dirichlet", "quantity", "domain", "language")
LORA_METHODS = ("adapter_average", "stack", "product_refactor")
DATASETS = ("synthetic", "ag_news", "multi_nli", "xnli")


class ConfigError(ValueError):
    """Raised when a configuration value is missing, unknown or out of range."""


DEFAULTS: dict[str, Any] = {
    "experiment": {"seed": 42, "rounds": 20, "num_clients": 10, "strategy": ["fedavg"]},
    "data": {
        "task": "classification",
        "dataset": "ag_news",
        "max_samples": 20000,
        "max_length": 128,
        "test_fraction": 0.2,
        "val_fraction": 0.1,
        "partition": {"mode": "dirichlet", "alpha": 0.3, "heterogeneity": 0.8},
    },
    "model": {"base": "distilbert-base-uncased",
              "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": None}},
    "train": {"local_epochs": 1, "lr": 2e-4, "batch_size": 16},
    "dropout": {"rate": 0.0, "min_participants": 2},
    "aggregation": {"lora_method": "adapter_average"},
    "grid": None,
}

_num = (int, float)
# key -> (accepted types, predicate, human-readable allowed range)
SCHEMA: dict[str, tuple[tuple, Any, str]] = {
    "experiment.seed": ((int,), lambda v: v >= 0, "integer >= 0"),
    "experiment.rounds": ((int,), lambda v: 1 <= v <= 1000, "integer in [1, 1000]"),
    "experiment.num_clients": ((int,), lambda v: 2 <= v <= 200, "integer in [2, 200]"),
    "experiment.strategy": ((list,), lambda v: len(v) > 0 and all(s in STRATEGIES for s in v),
                            f"non-empty list drawn from {list(STRATEGIES)}"),
    "data.task": ((str,), lambda v: v in ("classification",), "'classification'"),
    "data.dataset": ((str,), lambda v: v in DATASETS, f"one of {list(DATASETS)}"),
    "data.max_samples": ((int, type(None)), lambda v: v is None or v >= 100,
                         "null (use all) or integer >= 100"),
    "data.max_length": ((int,), lambda v: 8 <= v <= 512, "integer in [8, 512]"),
    "data.test_fraction": (_num, lambda v: 0 < v < 0.5, "number in (0, 0.5)"),
    "data.val_fraction": (_num, lambda v: 0 < v < 0.5, "number in (0, 0.5)"),
    "data.partition.mode": ((str,), lambda v: v in PARTITION_MODES, f"one of {list(PARTITION_MODES)}"),
    "data.partition.alpha": (_num, lambda v: v > 0, "number > 0 (smaller = more skew)"),
    "data.partition.heterogeneity": (_num, lambda v: 0 <= v <= 1, "number in [0, 1]"),
    "model.base": ((str,), lambda v: len(v) > 0, "non-empty model id"),
    "model.lora.rank": ((int,), lambda v: 1 <= v <= 256, "integer in [1, 256]"),
    "model.lora.alpha": (_num, lambda v: v > 0, "number > 0"),
    "model.lora.dropout": (_num, lambda v: 0 <= v < 1, "number in [0, 1)"),
    "model.lora.target_modules": ((list, type(None)),
                                  lambda v: v is None or (len(v) > 0 and all(isinstance(x, str) for x in v)),
                                  "null (auto-detect) or non-empty list of module names"),
    "train.local_epochs": ((int,), lambda v: 1 <= v <= 50, "integer in [1, 50]"),
    "train.lr": (_num, lambda v: 0 < v <= 1, "number in (0, 1]"),
    "train.batch_size": ((int,), lambda v: 1 <= v <= 1024, "integer in [1, 1024]"),
    "dropout.rate": (_num, lambda v: 0 <= v <= 1, "number in [0, 1]"),
    "dropout.min_participants": ((int,), lambda v: v >= 1, "integer >= 1"),
    "aggregation.lora_method": ((str,), lambda v: v in LORA_METHODS, f"one of {list(LORA_METHODS)}"),
}


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _get(cfg: dict, path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _set(cfg: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
        if not isinstance(cur, dict):
            raise ConfigError(f"{path}: '{part}' is not a section")
    cur[parts[-1]] = value


def _leaf_paths(d: dict, prefix: str = "") -> list[str]:
    paths = []
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            paths += _leaf_paths(v, p + ".")
        else:
            paths.append(p)
    return paths


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply 'section.key=value' strings (values parsed as YAML). Returns a new dict."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override '{item}' must look like section.key=value")
        key, raw = item.split("=", 1)
        _set(out, key.strip(), yaml.safe_load(raw))
    return out


def validate(cfg: dict) -> dict:
    """Check every key. Returns cfg unchanged if valid, raises ConfigError otherwise."""
    known = set(SCHEMA) | {"grid"}
    for path in _leaf_paths({k: v for k, v in cfg.items() if k != "grid"}):
        if path not in known:
            raise ConfigError(f"{path}: unknown key (typo?). Allowed keys: {sorted(known)}")
    for path, (types, ok, allowed) in SCHEMA.items():
        try:
            value = _get(cfg, path)
        except KeyError:
            raise ConfigError(f"{path}: missing. Allowed: {allowed}")
        if isinstance(value, bool) or not isinstance(value, types):
            raise ConfigError(f"{path}: wrong type {type(value).__name__}. Allowed: {allowed}")
        if not ok(value):
            raise ConfigError(f"{path}: invalid value {value!r}. Allowed: {allowed}")
    # cross-key rules
    n, m = cfg["experiment"]["num_clients"], cfg["dropout"]["min_participants"]
    if m > n:
        raise ConfigError(f"dropout.min_participants: {m} exceeds experiment.num_clients ({n}). Allowed: integer in [1, {n}]")
    grid = cfg.get("grid")
    if grid is not None:
        valid_grid_keys = {"alpha", "heterogeneity", "dropout_rate", "seeds"}
        if not isinstance(grid, dict) or not grid or set(grid) - valid_grid_keys:
            raise ConfigError(f"grid: must be a non-empty mapping with keys from {sorted(valid_grid_keys)}")
        for k, vals in grid.items():
            if not isinstance(vals, list) or not vals:
                raise ConfigError(f"grid.{k}: must be a non-empty list")
        for r in grid.get("dropout_rate", []):
            if not isinstance(r, _num) or not 0 <= r <= 1:
                raise ConfigError(f"grid.dropout_rate: invalid value {r!r}. Allowed: numbers in [0, 1]")
        for a in grid.get("alpha", []):
            if not isinstance(a, _num) or a <= 0:
                raise ConfigError(f"grid.alpha: invalid value {a!r}. Allowed: numbers > 0")
    return cfg


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    """defaults <- YAML file <- command-line overrides, then validated."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        if not isinstance(user, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        cfg = _deep_merge(cfg, user)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return validate(cfg)
