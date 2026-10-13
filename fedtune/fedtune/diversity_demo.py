"""See how diversity-aware weights differ from FedAvg on synthetic data (torch-free)."""
import numpy as np

from .aggregation.base import AggContext, ClientUpdate
from .aggregation.diversity import DiversityAware
from .aggregation.fedavg import FedAvg
from .client_summary import summarise
from .data.partition import build_partition
from .seeding import stream


def diversity_demo(cfg: dict, classes: int = 4, dim: int = 16) -> None:
    rng = np.random.default_rng(0)
    n = cfg["data"]["max_samples"] or 6000
    labels, groups = rng.integers(0, classes, n), rng.integers(0, 4, n)
    means = rng.normal(size=(classes, dim)); means /= np.linalg.norm(means, axis=1, keepdims=True)
    emb = 2.0 * means[labels] + 0.5 * rng.normal(size=(n, dim))        # class-structured fake "embeddings"
    part = build_partition(labels, cfg, groups=groups)
    K, seed = cfg["diversity"]["clusters"], cfg["experiment"]["seed"]
    sums = {c: summarise(emb[idx], K, stream(seed, "summary", c)) for c, idx in enumerate(part.client_train)}
    ups = [ClientUpdate(c, {}, len(idx)) for c, idx in enumerate(part.client_train)]
    ctx = AggContext(summaries=sums, n_clients=len(ups))
    agg = DiversityAware(cfg)
    w_div, w_avg = agg.weights(ups, ctx), FedAvg().weights(ups)
    gain = agg.info()["coverage_gain"]
    d = cfg["diversity"]
    print(f"clients={len(ups)} alpha={cfg['data']['partition']['alpha']} beta={d['beta']} gamma={d['gamma']} "
          f"universe={d['universe']} gain={d['gain']} max_weight={d['max_weight']} summary={next(iter(sums.values())).nbytes} bytes each\n")
    print(f"{'client':>6} {'n':>6}  " + " ".join(f"cls{c:>1}%" for c in range(classes)) +
          f"  {'gain':>6} {'FedAvg w':>9} {'diversity w':>12}")
    for c, idx in enumerate(part.client_train):
        share = np.bincount(labels[idx], minlength=classes) / len(idx) * 100
        print(f"{c:>6} {len(idx):>6}  " + " ".join(f"{x:>5.0f}" for x in share) +
              f"  {gain[str(c)]:>6.2f} {w_avg[c]:>9.3f} {w_div[c]:>12.3f}")
