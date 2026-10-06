import numpy as np

from .base import Aggregator, normalise


class FedAvg(Aggregator):
    """Baseline (SRS FR-13): weight each client by its share of the training data,
    w_k = n_k / sum(n)."""
    name = "fedavg"

    def weights(self, updates, ctx=None):
        return normalise(np.array([u.n_samples for u in updates], dtype=float))
