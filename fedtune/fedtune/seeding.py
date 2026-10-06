"""Named random streams.

One experiment seed feeds several *independent* random number generators
(one per purpose). Independence matters: changing the aggregation strategy
or the model must never change the data split or who drops out, otherwise
strategy comparisons would not be fair (SRS FR-3, FR-17).
"""
import numpy as np

# Fixed integer id per stream. Never renumber: it would change every old result.
STREAMS = {"partition": 1, "dropout": 2, "model_init": 3, "training": 4, "summary": 5, "data": 6}


def stream(seed: int, name: str, *keys: int) -> np.random.Generator:
    """Return a reproducible generator for (seed, purpose[, extra integer keys such as
    round and client id])."""
    if name not in STREAMS:
        raise KeyError(f"unknown random stream '{name}', choose from {sorted(STREAMS)}")
    return np.random.default_rng(np.random.SeedSequence([int(seed), STREAMS[name], *[int(k) for k in keys]]))
