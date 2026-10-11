"""The federated round loop (SRS UC3, UC4, UC5, UC8, UC12)."""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from .aggregation import get_aggregator
from .aggregation.base import ClientUpdate
from .client import train_client
from .evaluate import evaluate
from .model import is_finite_state, set_trainable_state, state_bytes_split, state_num_bytes
from .participation import ParticipationModel
from .pipeline import World, build_world
from .runlog import RunLogger, build_summary, make_run_id

_KEEP = ("accuracy", "macro_f1", "loss")


def _m(d: dict) -> dict:
    return {k: d[k] for k in _KEEP}


def evaluate_global(world: World, state: dict):
    """Global model on the global test set, and on every client's own validation set."""
    bs = world.cfg["train"]["batch_size"]
    set_trainable_state(world.model, state)
    g = evaluate(world.model, world.enc, world.partition.test_idx, bs, world.device)
    per_client = {str(c): _m(evaluate(world.model, world.enc, v, bs, world.device))
                  for c, v in enumerate(world.partition.client_val)}
    return _m(g), per_client


def run_strategy(world: World, strategy: str, out_dir: str, verbose: bool = True, fault_hook=None) -> dict:
    """fault_hook(round, client_id, state) -> state is for tests: it can corrupt an update."""
    cfg = world.cfg
    aggregator = get_aggregator(strategy)  # fails early if not implemented
    run_id = make_run_id(cfg, strategy)
    rounds, m_min = cfg["experiment"]["rounds"], cfg["dropout"]["min_participants"]
    if world.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    global_state = {k: v.clone() for k, v in world.init_state.items()}
    payload = state_num_bytes(global_state)
    pm = ParticipationModel(cfg, [len(s) for s in world.partition.client_train], payload)
    cum_up = cum_down = 0
    sim_cum = 0.0
    records = []

    with RunLogger(os.path.join(out_dir, run_id)) as log:
        log.write_json("config.json", cfg)  # alone enough to regenerate partition and participation
        log.write_json("meta.json", {"run_id": run_id, "strategy": strategy})
        log.write_json("partition_report.json", world.partition.report)
        np.save(os.path.join(log.dir, "participation.npy"), pm.matrix())
        try:
            g, pc = evaluate_global(world, global_state)
            records.append({"run_id": run_id, "strategy": strategy, "round": 0, "skipped": False,
                            "skip_reason": None, "selected_clients": [], "participating_clients": [],
                            "dropped_clients": [], "late_clients": [], "excluded_clients": [], "cause": {},
                            "client_times": {}, "aggregation_weights": {}, "clusters": None,
                            "global_metrics": g, "per_client": pc, "client_train_loss": {},
                            "comm_bytes": {"up": 0, "down": 0, "up_lora": 0, "up_head": 0},
                            "comm_cum_bytes": {"up": 0, "down": 0}, "round_time_s": 0.0,
                            "sim_time_s": 0.0, "sim_time_cum_s": 0.0})
            log.log_round(records[0])
            if verbose:
                print(f"[{run_id}] round 0/{rounds}  acc={g['accuracy']:.3f} f1={g['macro_f1']:.3f}")

            for r in range(1, rounds + 1):
                t0 = time.time()
                sel = pm.select(r)
                updates, losses, excluded = [], {}, []
                up_lora = up_head = 0
                skip_reason = "too_few_participants" if sel.skipped else None
                if not sel.skipped:
                    for cid in sel.participants:
                        res = train_client(world.model, global_state, world.enc, world.partition.client_train[cid],
                                           cfg, r, cid, world.device)
                        state = fault_hook(r, cid, res.state) if fault_hook else res.state
                        lo, hd = state_bytes_split(state)
                        up_lora, up_head = up_lora + lo, up_head + hd      # the update was transmitted either way
                        losses[str(cid)] = res.mean_loss
                        if is_finite_state(state):
                            updates.append(ClientUpdate(cid, state, res.n_samples))
                        else:
                            excluded.append({"client": cid, "reason": "non_finite"})
                    if len(updates) < m_min:
                        skip_reason = "too_few_valid_updates"
                skipped = skip_reason is not None
                weights, clusters = {}, None
                if not skipped:
                    result = aggregator.aggregate(global_state, updates)
                    global_state = result.state
                    weights = {str(k): v for k, v in result.weights.items()}
                    clusters = result.clusters
                    g, pc = evaluate_global(world, global_state)
                # a skipped round leaves the global model, and therefore its metrics, unchanged
                down = payload * len(sel.selected)
                up = up_lora + up_head
                cum_up, cum_down = cum_up + up, cum_down + down
                sim_cum += sel.duration_s
                cause = dict(sel.cause)
                for e in excluded:
                    cause[e["client"]] = "non_finite"
                rec = {"run_id": run_id, "strategy": strategy, "round": r, "skipped": skipped,
                       "skip_reason": skip_reason, "selected_clients": sel.selected,
                       "participating_clients": [] if skipped else [u.client_id for u in updates],
                       "dropped_clients": sel.dropped, "late_clients": sel.late, "excluded_clients": excluded,
                       "cause": {str(k): v for k, v in cause.items()},
                       "client_times": {str(k): v for k, v in sel.times.items()},
                       "aggregation_weights": weights, "clusters": clusters,
                       "global_metrics": g, "per_client": pc, "client_train_loss": losses,
                       "comm_bytes": {"up": up, "down": down, "up_lora": up_lora, "up_head": up_head},
                       "comm_cum_bytes": {"up": cum_up, "down": cum_down},
                       "round_time_s": time.time() - t0, "sim_time_s": sel.duration_s, "sim_time_cum_s": sim_cum}
                records.append(rec)
                log.log_round(rec)
                if verbose:
                    note = f"  SKIPPED ({skip_reason})" if skipped else ""
                    print(f"[{run_id}] round {r}/{rounds}  acc={g['accuracy']:.3f} f1={g['macro_f1']:.3f}  "
                          f"ok={len(rec['participating_clients'])} drop={len(sel.dropped)} late={len(sel.late)} "
                          f"bad={len(excluded)}  up={up / 1e6:.2f}MB  sim={sel.duration_s:.1f}s  "
                          f"{rec['round_time_s']:.1f}s{note}")
        except BaseException as e:  # keep logs readable and record the failure (SRS UC8, FR-34)
            log.write_json("summary.json", {"run_id": run_id, "strategy": strategy, "status": "failed",
                                            "error": f"{type(e).__name__}: {e}", "rounds_logged": len(records) - 1})
            raise
        peak = torch.cuda.max_memory_allocated() / 1e6 if world.device.type == "cuda" else None
        summary = build_summary(records, run_id, strategy, time.time() - t_start, peak)
        log.write_json("summary.json", summary)
    return summary


def run_all(cfg: dict, out_dir: str = "runs") -> list:
    """Run every strategy in the config on the SAME partition, seed, participation and starting model (UC5)."""
    strategies = cfg["experiment"]["strategy"]
    for s in strategies:
        get_aggregator(s)  # fail before any slow work
    world = build_world(cfg)
    summaries = []
    for s in strategies:
        set_trainable_state(world.model, world.init_state)
        summaries.append(run_strategy(world, s, out_dir))
    print(f"\n{'strategy':<12} {'final acc':>10} {'final F1':>9} {'worst client':>13} {'skipped':>8} {'upload MB':>10} {'sim s':>8} {'time s':>8}")
    for s in summaries:
        d, c = s["per_client_accuracy_dispersion"], s["cumulative_comm_bytes"]
        print(f"{s['strategy']:<12} {s['final_metrics']['accuracy']:>10.3f} {s['final_metrics']['macro_f1']:>9.3f} "
              f"{d['min']:>13.3f} {s['rounds_skipped']:>8} {c['up'] / 1e6:>10.2f} {s['simulated_time_s']:>8.1f} {s['total_time_s']:>8.1f}")
    print(f"logs written under {out_dir}/")
    return summaries
