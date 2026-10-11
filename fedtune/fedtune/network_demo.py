"""Explore dropout and network effects without training anything (torch-free)."""
import numpy as np

from .data.partition import build_partition
from .participation import ParticipationModel


def network_demo(cfg: dict, payload_mb: float = 2.964) -> None:
    rng = np.random.default_rng(0)                       # fixed synthetic data, only the shard sizes matter
    n = cfg["data"]["max_samples"] or 8000
    part = build_partition(rng.integers(0, 4, n), cfg, groups=rng.integers(0, 4, n))
    sizes = [len(s) for s in part.client_train]
    pm = ParticipationModel(cfg, sizes, int(payload_mb * 1e6))
    D = cfg["network"]["deadline_s"]
    print(f"clients={pm.K} rounds={pm.R} dropout p={pm.p} m_min={pm.m_min} payload={payload_mb} MB "
          f"deadline={'none' if D is None else str(D) + ' s'} network={cfg['network']['profile']}")
    if pm.profiles:
        print(f"\n{'client':>6} {'shard':>6} {'tier':>11} {'up Mb/s':>8} {'down Mb/s':>9} {'lat ms':>7} {'slow':>5} "
              f"{'steps':>6} {'down s':>7} {'comp s':>7} {'up s':>7} {'total s':>8}")
        for k in range(pm.K):
            p, t = pm.profiles[k], pm.select(1).times[k]
            print(f"{k:>6} {sizes[k]:>6} {p.tier:>11} {p.up_mbps:>8.1f} {p.down_mbps:>9.1f} {p.latency_s * 1000:>7.0f} "
                  f"{p.slowdown:>5.1f} {pm.steps[k]:>6} {t['down']:>7.1f} {t['compute']:>7.1f} {t['up']:>7.1f} {t['total']:>8.1f}")
    print(f"\n{'round':>5} {'ok':>3} {'dropped':>8} {'late':>5} {'skipped':>8} {'sim s':>7}   participants")
    tot = {"ok": 0, "drop": 0, "late": 0, "skip": 0}
    sim = 0.0
    for r in range(1, pm.R + 1):
        s = pm.select(r)
        sim += s.duration_s
        tot["ok"] += len(s.participants); tot["drop"] += len(s.dropped); tot["late"] += len(s.late); tot["skip"] += s.skipped
        print(f"{r:>5} {len(s.participants):>3} {len(s.dropped):>8} {len(s.late):>5} {str(s.skipped):>8} {s.duration_s:>7.1f}   {s.participants}")
    print(f"\ntotals: on-time updates={tot['ok']} dropouts={tot['drop']} deadline misses={tot['late']} "
          f"skipped rounds={tot['skip']}  simulated time={sim:.1f} s")
