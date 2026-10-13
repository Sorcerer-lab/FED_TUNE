"""Diversity-aware aggregation (proposed strategy 1).

Weight of client k:  w_k  proportional to  n_k^beta * (c_k + epsilon)^gamma
where c_k is the client's *coverage gain*: how much of the data space (as seen through the client
summaries) would be lost if this client were left out, divided by n_k ("per_sample", default) so that
volume is not counted twice, and scaled to mean 1. gain="total" skips the division (ablation).
beta=1, gamma=0 is exactly FedAvg.
"""
from __future__ import annotations

import numpy as np

from .base import Aggregator, normalise


def coverage_gains(summaries: dict, participants: list, universe_ids: list) -> dict:
    """Leave-one-out coverage gain of every participant.

    f(S) = sum over pooled centroids u of m_u * max_{s in centroids(S)} cos(u, s), with U the centroids
    of the clients in `universe_ids` and S the participants. gain_k = f(S) - f(S without k).
    """
    U = np.concatenate([summaries[c].centroids for c in universe_ids]).astype(np.float64)
    m = np.concatenate([summaries[c].counts for c in universe_ids]).astype(np.float64)
    S = np.concatenate([summaries[c].centroids for c in participants]).astype(np.float64)
    owner = np.concatenate([[c] * len(summaries[c].counts) for c in participants])
    sim = U @ S.T                                                     # (|U|, |S|) cosine similarities

    def f(mask):
        return float((m * sim[:, mask].max(axis=1)).sum()) if mask.any() else 0.0

    full = f(np.ones(len(owner), dtype=bool))
    return {c: full - f(owner != c) for c in participants}


def cap_weights(w: np.ndarray, max_w: float | None) -> np.ndarray:
    """No weight above max_w (raised to 1/m if the cap is infeasible); the surplus is shared proportionally."""
    w = np.asarray(w, dtype=float)
    if max_w is None:
        return w
    cap = max(max_w, 1.0 / len(w))
    for _ in range(len(w) + 1):
        over = w > cap + 1e-12
        if not over.any():
            break
        free = ~over
        surplus = w[over].sum() - cap * over.sum()
        w = np.where(over, cap, w)
        if free.any() and w[free].sum() > 0:
            w[free] += surplus * w[free] / w[free].sum()
    return w


class DiversityAware(Aggregator):
    name = "diversity"
    needs_summary = True

    def __init__(self, cfg: dict | None = None):
        from ..config import DEFAULTS
        d = (cfg or DEFAULTS)["diversity"]
        self.beta, self.gamma, self.eps = float(d["beta"]), float(d["gamma"]), float(d["epsilon"])
        self.max_weight, self.universe = d["max_weight"], d["universe"]
        self.gain_norm = d.get("gain", "per_sample")
        self._info = None

    def weights(self, updates, ctx=None):
        if ctx is None or not ctx.summaries:
            raise ValueError("diversity-aware aggregation needs client summaries (none were provided)")
        ids = [u.client_id for u in updates]
        n = np.array([u.n_samples for u in updates], dtype=float)
        if len(ids) == 1:
            self._info = {"coverage_gain": {str(ids[0]): 1.0}, "universe": self.universe}
            return np.array([1.0])
        universe = ids if self.universe == "participants" else sorted(ctx.summaries)
        gain = coverage_gains(ctx.summaries, ids, universe)
        raw = np.array([gain[i] for i in ids])
        c = raw / n if self.gain_norm == "per_sample" else raw.copy()
        c = c / c.mean() if c.mean() > 1e-12 else np.ones_like(c)         # scale to mean 1
        w = normalise(n ** self.beta * (c + self.eps) ** self.gamma)
        w = normalise(cap_weights(w, self.max_weight))
        self._info = {"coverage_gain": {str(i): float(x) for i, x in zip(ids, c)},
                      "gain_raw": {str(i): float(x) for i, x in zip(ids, raw)},
                      "universe": self.universe, "gain": self.gain_norm}
        return w

    def info(self):
        return self._info
