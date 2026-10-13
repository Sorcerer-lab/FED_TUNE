"""Bounded client data summaries (no torch needed).

A client embeds its training texts with the frozen base encoder, groups the embeddings with k-means
and sends only the K centroids and their counts. No text, no token ids and no per-example labels
leave the client; what the summary does reveal (approximate cluster centres of the client's data
in embedding space) is documented in summary_log.json.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClientSummary:
    centroids: np.ndarray    # (K, d) float32, unit length
    counts: np.ndarray       # (K,) int32, samples assigned to each centroid

    @property
    def nbytes(self) -> int:
        return int(self.centroids.nbytes + self.counts.nbytes)


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def kmeans(X: np.ndarray, K: int, rng: np.random.Generator, iters: int = 30):
    """Plain Lloyd k-means with k-means++ seeding. Returns (centers (K,d), assignment (n,))."""
    n = len(X)
    K = max(1, min(K, n))
    centers = [X[rng.integers(n)]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, K):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(X[rng.choice(n, p=p)])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(1))
    C = np.array(centers, dtype=np.float64)
    assign = np.zeros(n, dtype=int)
    for it in range(iters):
        dist = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        new = dist.argmin(1)
        if it > 0 and np.array_equal(new, assign):
            break
        assign = new
        for k in range(K):
            m = assign == k
            if m.any():
                C[k] = X[m].mean(0)
    return C, assign


def summarise(embeddings: np.ndarray, K: int, rng: np.random.Generator) -> ClientSummary:
    X = _unit(np.asarray(embeddings, dtype=np.float64))
    C, assign = kmeans(X, K, rng)
    counts = np.bincount(assign, minlength=len(C))
    keep = counts > 0                                   # drop empty cells: they carry no information
    return ClientSummary(_unit(C[keep]).astype(np.float32), counts[keep].astype(np.int32))


def save_summaries(path: str, summaries: dict) -> None:
    arrs = {}
    for cid, s in summaries.items():
        arrs[f"c{cid}_centroids"], arrs[f"c{cid}_counts"] = s.centroids, s.counts
    np.savez(path, **arrs)


def load_summaries(path: str) -> dict:
    z = np.load(path)
    ids = sorted({int(k[1:].split("_")[0]) for k in z.files})
    return {i: ClientSummary(z[f"c{i}_centroids"], z[f"c{i}_counts"]) for i in ids}


def summary_report(summaries: dict) -> dict:
    """What goes into summary_log.json (SRS FR-12): sizes and exactly what the summary reveals."""
    return {
        "reveals": "K unit-length centroids of the client's text embeddings (frozen base encoder) and how many "
                   "samples fall in each; no text, no token ids, no per-example labels",
        "clients": {str(c): {"centroids": int(len(s.counts)), "dim": int(s.centroids.shape[1]),
                             "samples": int(s.counts.sum()), "bytes": s.nbytes} for c, s in summaries.items()},
        "total_bytes": int(sum(s.nbytes for s in summaries.values())),
    }
