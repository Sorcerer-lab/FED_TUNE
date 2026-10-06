"""Non-IID data partitioning (SRS UC2, FR-5, FR-6, FR-7).

Works on *indices only* (labels and optional group attributes), never raw text,
so the partition report can never leak document content.

Modes
  dirichlet : label skew. Per class, split its samples across clients with
              proportions drawn from Dirichlet(alpha). Small alpha = strong skew.
  quantity  : quantity skew. Labels stay IID, but client SIZES are unequal
              (sizes drawn from Dirichlet(alpha)).
  domain    : each sample has a group (topic/domain). Client c has a "home"
              group; a fraction `heterogeneity` of each group's samples goes to
              its home clients, the rest is spread uniformly.
  language  : identical mechanics to 'domain', with language as the group.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..seeding import stream

MIN_CLIENT_SIZE = 10  # train+val samples every client must hold


class PartitionError(RuntimeError):
    pass


@dataclass
class Partition:
    test_idx: np.ndarray
    client_train: list[np.ndarray]
    client_val: list[np.ndarray]
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------- measures
def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, log base 2, so the value lies in [0, 1]."""
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _distribution(values: np.ndarray, categories: np.ndarray) -> np.ndarray:
    counts = np.array([(values == c).sum() for c in categories], dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def heterogeneity_score(values: np.ndarray, shards: list[np.ndarray]) -> float:
    """Size-weighted mean JS divergence between each client's distribution of
    `values` (labels or groups) and the global distribution. 0 = IID, 1 = maximal."""
    cats = np.unique(values)
    global_dist = _distribution(values, cats)
    sizes = np.array([len(s) for s in shards], dtype=float)
    scores = np.array([_js_divergence(_distribution(values[s], cats), global_dist) for s in shards])
    return float((scores * sizes).sum() / sizes.sum())


# ---------------------------------------------------------------- splitters
def _split_by_proportions(idx: np.ndarray, p: np.ndarray) -> list[np.ndarray]:
    cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
    return np.split(idx, cuts)


def _dirichlet_label_skew(labels, n_clients, alpha, rng, min_size, max_tries=200):
    classes = np.unique(labels)
    for _ in range(max_tries):
        shards = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.flatnonzero(labels == c)
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(n_clients, alpha))
            for cid, part in enumerate(_split_by_proportions(idx, p)):
                shards[cid].extend(part.tolist())
        if min(len(s) for s in shards) >= min_size:
            return [np.array(s, dtype=int) for s in shards]
    raise PartitionError(
        f"could not give every client >= {min_size} samples with alpha={alpha} after {max_tries} tries; "
        "raise data.partition.alpha or use fewer clients"
    )


def _quantity_skew(n, n_clients, alpha, rng, min_size, max_tries=200):
    for _ in range(max_tries):
        idx = rng.permutation(n)
        p = rng.dirichlet(np.full(n_clients, alpha))
        shards = _split_by_proportions(idx, p)
        if min(len(s) for s in shards) >= min_size:
            return [np.asarray(s, dtype=int) for s in shards]
    raise PartitionError(
        f"could not give every client >= {min_size} samples with alpha={alpha}; raise data.partition.alpha"
    )


def _group_skew(groups, n_clients, heterogeneity, rng):
    """Home-group assignment: client c's home group is group (c mod G)."""
    cats = np.unique(groups)
    G = len(cats)
    homes = {}
    for gi, g in enumerate(cats):
        h = [c for c in range(n_clients) if c % G == gi]
        homes[g] = h if h else [gi % n_clients]
    shards = [[] for _ in range(n_clients)]
    for g in cats:
        idx = np.flatnonzero(groups == g)
        use_home = rng.random(len(idx)) < heterogeneity
        home_pick = rng.choice(homes[g], size=len(idx))
        any_pick = rng.integers(0, n_clients, size=len(idx))
        for i, uh, hp, ap in zip(idx, use_home, home_pick, any_pick):
            shards[hp if uh else ap].append(int(i))
    return [np.array(s, dtype=int) for s in shards]


# ---------------------------------------------------------------- main entry
def build_partition(labels, cfg: dict, groups=None) -> Partition:
    """Create global test set, per-client train shards and per-client validation sets."""
    labels = np.asarray(labels)
    n = len(labels)
    seed = cfg["experiment"]["seed"]
    n_clients = cfg["experiment"]["num_clients"]
    pcfg = cfg["data"]["partition"]
    mode = pcfg["mode"]
    rng = stream(seed, "partition")

    if mode in ("domain", "language") and groups is None:
        raise PartitionError(f"mode '{mode}' needs a per-sample '{mode}' attribute (FR-4); none was supplied")
    groups = None if groups is None else np.asarray(groups)

    # 1) global test set (FR-7): stratified so every class appears in the test set
    test_frac = cfg["data"]["test_fraction"]
    test_parts, pool_parts = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_frac)))
        test_parts.append(idx[:k])
        pool_parts.append(idx[k:])
    test_idx = np.sort(np.concatenate(test_parts))
    pool = np.sort(np.concatenate(pool_parts))

    # 2) split the remaining pool across clients (positions refer to `pool`)
    pool_labels = labels[pool]
    min_size = MIN_CLIENT_SIZE
    if mode == "dirichlet":
        pos = _dirichlet_label_skew(pool_labels, n_clients, pcfg["alpha"], rng, min_size)
    elif mode == "quantity":
        pos = _quantity_skew(len(pool), n_clients, pcfg["alpha"], rng, min_size)
    else:
        pos = _group_skew(groups[pool], n_clients, pcfg["heterogeneity"], rng)
        if min(len(s) for s in pos) < min_size:
            raise PartitionError("a client received too few samples; lower heterogeneity or use fewer clients")
    shards = [pool[p] for p in pos]

    # 3) per-client validation split (FR-7)
    val_frac = cfg["data"]["val_fraction"]
    client_train, client_val = [], []
    for s in shards:
        s = rng.permutation(s)
        k = max(1, int(round(len(s) * val_frac)))
        client_val.append(np.sort(s[:k]))
        client_train.append(np.sort(s[k:]))

    report = make_report(labels, groups, client_train, client_val, test_idx, cfg)
    return Partition(test_idx, client_train, client_val, report)


def make_report(labels, groups, client_train, client_val, test_idx, cfg) -> dict:
    """Partition report (FR-6): sizes, distributions, numeric heterogeneity. No raw text."""
    cats = np.unique(labels)
    clients = []
    for cid, (tr, va) in enumerate(zip(client_train, client_val)):
        entry = {
            "client": cid,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "label_counts": {str(c): int((labels[tr] == c).sum()) for c in cats},
        }
        if groups is not None:
            entry["group_counts"] = {str(g): int((groups[tr] == g).sum()) for g in np.unique(groups)}
        clients.append(entry)
    report = {
        "seed": cfg["experiment"]["seed"],
        "mode": cfg["data"]["partition"]["mode"],
        "alpha": cfg["data"]["partition"]["alpha"],
        "heterogeneity_param": cfg["data"]["partition"]["heterogeneity"],
        "n_clients": len(client_train),
        "n_test": int(len(test_idx)),
        "label_heterogeneity_js": heterogeneity_score(labels, client_train),
        "size_min": int(min(len(t) for t in client_train)),
        "size_max": int(max(len(t) for t in client_train)),
        "clients": clients,
    }
    if groups is not None:
        report["group_heterogeneity_js"] = heterogeneity_score(groups, client_train)
    return report


def write_report(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
