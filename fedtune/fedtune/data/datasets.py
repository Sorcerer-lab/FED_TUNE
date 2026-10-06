"""Dataset loading (SRS FR-4). Returns plain python/numpy objects: texts, labels and an
optional per-sample group (topic/domain/language) used for partitioning.

  synthetic : offline generated data (for tests and the tiny check)
  ag_news   : news topic classification, 4 classes            (label skew)
  multi_nli : sentence-pair inference, 3 classes; the genre   (domain skew)
              of each pair (fiction, government, ...) is the group
  xnli      : the same inference task in several languages;   (language skew)
              the language is the group
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..seeding import stream

XNLI_LANGS = ["en", "fr", "de", "es", "hi"]
NLI_LABELS = ["entailment", "neutral", "contradiction"]


class DataError(RuntimeError):
    pass


@dataclass
class Bundle:
    texts: list
    pairs: list | None        # second sentence for pair tasks (NLI), else None
    labels: np.ndarray
    groups: np.ndarray | None
    label_names: list
    group_names: list | None


def _subsample(n: int, k, rng) -> np.ndarray:
    if k is None or n <= k:
        return np.arange(n)
    return np.sort(rng.choice(n, size=k, replace=False))


def _load_hf(ids, *args, **kwargs):
    """Try several Hugging Face dataset ids (they were renamed over time)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DataError("the 'datasets' package is missing: pip install datasets") from e
    errors = []
    for i in ids:
        try:
            return load_dataset(i, *args, **kwargs)
        except Exception as e:  # network, name or schema problems
            errors.append(f"{i}: {type(e).__name__}: {e}")
    raise DataError("could not load the dataset:\n  " + "\n  ".join(errors))


def _synthetic(n: int, seed: int, n_classes: int = 4, n_groups: int = 4) -> Bundle:
    rng = stream(seed, "data")
    common = [f"w{i}" for i in range(40)]
    class_words = [[f"c{c}_{i}" for i in range(15)] for c in range(n_classes)]
    group_words = [[f"g{g}_{i}" for i in range(8)] for g in range(n_groups)]
    labels = rng.integers(0, n_classes, n)
    groups = rng.integers(0, n_groups, n)
    texts = []
    for y, g in zip(labels, groups):
        words = ([str(w) for w in rng.choice(common, 5)]
                 + [str(w) for w in rng.choice(class_words[y], 6)]
                 + [str(w) for w in rng.choice(group_words[g], 3)])
        rng.shuffle(words)
        texts.append(" ".join(words))
    return Bundle(texts, None, labels, groups,
                  [f"class{c}" for c in range(n_classes)], [f"group{g}" for g in range(n_groups)])


def _ag_news(k, seed) -> Bundle:
    ds = _load_hf(["fancyzhx/ag_news", "ag_news"], split="train")
    ds = ds.select(_subsample(len(ds), k, stream(seed, "data")).tolist())
    return Bundle(list(ds["text"]), None, np.array(ds["label"]), None,
                  ["World", "Sports", "Business", "Sci/Tech"], None)


def _multi_nli(k, seed) -> Bundle:
    ds = _load_hf(["nyu-mll/multi_nli", "multi_nli"], split="train")
    ds = ds.select(_subsample(len(ds), k, stream(seed, "data")).tolist())
    genres = list(ds["genre"])
    names = sorted(set(genres))
    groups = np.array([names.index(g) for g in genres])
    return Bundle(list(ds["premise"]), list(ds["hypothesis"]), np.array(ds["label"]),
                  groups, NLI_LABELS, names)


def _xnli(k, seed) -> Bundle:
    per_lang = None if k is None else max(1, k // len(XNLI_LANGS))
    texts, pairs, labels, groups = [], [], [], []
    for gi, lang in enumerate(XNLI_LANGS):
        ds = _load_hf(["facebook/xnli", "xnli"], lang, split="train")
        ds = ds.select(_subsample(len(ds), per_lang, stream(seed, "data", gi)).tolist())
        texts += list(ds["premise"])
        pairs += list(ds["hypothesis"])
        labels += list(ds["label"])
        groups += [gi] * len(ds)
    return Bundle(texts, pairs, np.array(labels), np.array(groups), NLI_LABELS, list(XNLI_LANGS))


def load_bundle(cfg: dict) -> Bundle:
    name = cfg["data"]["dataset"]
    k = cfg["data"]["max_samples"]
    seed = cfg["experiment"]["seed"]
    if name == "synthetic":
        return _synthetic(k or 2000, seed)
    if name == "ag_news":
        return _ag_news(k, seed)
    if name == "multi_nli":
        return _multi_nli(k, seed)
    if name == "xnli":
        return _xnli(k, seed)
    raise DataError(f"unknown dataset '{name}'")
