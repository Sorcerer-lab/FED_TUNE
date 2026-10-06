"""Aggregation building blocks. No torch import here, so the maths can be unit-tested anywhere:
a "state" is a dict {parameter name: array-like}, and works for torch tensors and numpy arrays."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClientUpdate:
    client_id: int
    state: dict              # the client's trainable parameters after local training
    n_samples: int           # size of its training shard
    summary: object = None   # bounded-size data summary (used from Step 6)


@dataclass
class AggregationResult:
    state: dict
    weights: dict            # client_id -> weight actually used (sums to 1)
    clusters: list | None = None


def normalise(w) -> np.ndarray:
    """Scale non-negative weights so they sum to 1 (FR-20: renormalise over participants)."""
    w = np.asarray(w, dtype=float)
    if w.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(f"weights must be finite, non-negative and not all zero, got {w.tolist()}")
    return w / w.sum()


def weighted_average(states: list, weights) -> dict:
    """Element-wise  sum_k w_k * state_k  over every named parameter."""
    keys = list(states[0].keys())
    for s in states[1:]:
        if list(s.keys()) != keys:
            raise ValueError("client states have different parameter names")
    out = {}
    for k in keys:
        acc = None
        for w, s in zip(weights, states):
            term = s[k] * float(w)
            acc = term if acc is None else acc + term
        out[k] = acc
    return out


class Aggregator:
    """Base class. A strategy only has to say how much each client counts (`weights`);
    strategies that need more (clustering) override `aggregate`."""
    name = "base"

    def weights(self, updates: list, ctx=None) -> np.ndarray:
        raise NotImplementedError

    def aggregate(self, global_state: dict, updates: list, ctx=None) -> AggregationResult:
        if not updates:
            raise ValueError("cannot aggregate zero updates")
        w = self.weights(updates, ctx)
        state = weighted_average([u.state for u in updates], w)
        return AggregationResult(state, {u.client_id: float(x) for u, x in zip(updates, w)})
