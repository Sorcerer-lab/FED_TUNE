"""Analytic network model (no torch needed).

Each client has an uplink and downlink rate, a one-way latency and a compute slowdown. The simulated
time of one round for client k is  download + local compute + upload  (see docs/04). Everything is
a formula of (seed, round, client, payload size), never of measured wall-clock time, so every
aggregation strategy sees the same participants.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .seeding import stream

TIERS = [  # (name, uplink Mbit/s, downlink Mbit/s, one-way latency s, compute slowdown)
    ("fast", 50.0, 100.0, 0.020, 1.0),
    ("medium", 10.0, 50.0, 0.060, 1.5),
    ("slow", 1.0, 10.0, 0.150, 3.0),
]


@dataclass(frozen=True)
class Profile:
    tier: str
    up_mbps: float
    down_mbps: float
    latency_s: float
    slowdown: float


def make_profiles(cfg: dict, shard_sizes) -> list | None:
    """One Profile per client, or None when the network model is off."""
    net = cfg["network"]
    if net["profile"] == "off":
        return None
    n = len(shard_sizes)
    if net["profile"] == "homogeneous":
        p = Profile("homogeneous", float(net["uplink_mbps"]), float(net["downlink_mbps"]),
                    net["latency_ms"] / 1000.0, float(net["slowdown"]))
        return [p] * n
    # tiered: clients are put in an order, the first third gets "fast", the last third "slow"
    sizes = np.asarray(shard_sizes)
    if net["assignment"] == "random":
        order = stream(cfg["experiment"]["seed"], "network", 0).permutation(n)
    elif net["assignment"] == "large_shards_slow":
        order = np.argsort(sizes, kind="stable")            # smallest shards first (fast)
    else:  # small_shards_slow
        order = np.argsort(-sizes, kind="stable")           # largest shards first (fast)
    tier_of = {}
    for pos, cid in enumerate(order):
        tier_of[int(cid)] = TIERS[min(2, int(3 * pos / n))]
    return [Profile(tier_of[c][0], tier_of[c][1], tier_of[c][2], tier_of[c][3], tier_of[c][4]) for c in range(n)]


def client_time(profile: Profile, payload_down: int, payload_up: int, steps: int, step_time_s: float,
                up_factor: float = 1.0, down_factor: float = 1.0) -> dict:
    """Simulated seconds for one client in one round. Payloads are in bytes, rates in Mbit/s."""
    down = profile.latency_s + 8.0 * payload_down / (profile.down_mbps * 1e6 * down_factor)
    compute = step_time_s * profile.slowdown * steps
    up = profile.latency_s + 8.0 * payload_up / (profile.up_mbps * 1e6 * up_factor)
    return {"down": down, "compute": compute, "up": up, "total": down + compute + up}


def local_steps(n_samples: int, batch_size: int, epochs: int) -> int:
    return int(math.ceil(n_samples / batch_size) * epochs)
