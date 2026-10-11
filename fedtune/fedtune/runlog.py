"""Run logging and the end-of-run summary (SRS FR-24, FR-25, 4.6). No torch needed."""
from __future__ import annotations

import json
import os

import numpy as np

from .metrics import dispersion


def make_run_id(cfg: dict, strategy: str) -> str:
    p = cfg["data"]["partition"]
    knob = p["alpha"] if p["mode"] in ("dirichlet", "quantity") else p["heterogeneity"]
    s = lambda x: str(x).replace(".", "p")  # noqa: E731
    return (f"{cfg['data']['dataset']}-{p['mode']}{s(knob)}-drop{s(cfg['dropout']['rate'])}"
            f"-{strategy}-s{cfg['experiment']['seed']}")


def _plain(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"cannot serialise {type(o)}")


class RunLogger:
    """Writes one JSON object per round, flushed immediately, so logs stay readable even if
    the run crashes (NFR-7)."""

    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.dir = run_dir
        self._f = open(os.path.join(run_dir, "rounds.jsonl"), "w", encoding="utf-8")

    def log_round(self, record: dict) -> None:
        self._f.write(json.dumps(record, default=_plain) + "\n")
        self._f.flush()

    def write_json(self, name: str, obj) -> None:
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=_plain)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()
        return False


def build_summary(records: list, run_id: str, strategy: str, total_time_s: float, peak_memory_mb) -> dict:
    """records[0] is the round-0 record (initial model, before any training)."""
    rounds = [r for r in records if r["round"] >= 1]
    done = [r for r in rounds if not r["skipped"]]
    last = records[-1]
    accs = [m["accuracy"] for m in last["per_client"].values()]
    n = lambda key: int(sum(len(r.get(key, [])) for r in rounds))  # noqa: E731
    reasons = {}
    for r in rounds:
        if r["skipped"]:
            k = r.get("skip_reason") or "unknown"
            reasons[k] = reasons.get(k, 0) + 1
    return {
        "run_id": run_id,
        "strategy": strategy,
        "status": "completed",
        "rounds_completed": len(done),
        "rounds_skipped": len(rounds) - len(done),
        "skip_reasons": reasons,
        "mean_participants_per_round": float(np.mean([len(r["participating_clients"]) for r in rounds])) if rounds else 0.0,
        "absences": {"dropout": n("dropped_clients"), "deadline": n("late_clients"), "non_finite": n("excluded_clients")},
        "initial_metrics": records[0]["global_metrics"],
        "final_metrics": last["global_metrics"],
        "per_client_accuracy_dispersion": dispersion(accs) if accs else None,
        "cumulative_comm_bytes": last["comm_cum_bytes"],
        "simulated_time_s": float(last.get("sim_time_cum_s", 0.0)),
        "total_time_s": total_time_s,
        "peak_memory_mb": peak_memory_mb,
    }
