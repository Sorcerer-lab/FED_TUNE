"""The federated round loop (SRS UC3, UC5)."""
from __future__ import annotations

import os
import time

import torch

from .aggregation import get_aggregator
from .aggregation.base import ClientUpdate
from .client import train_client
from .evaluate import evaluate
from .model import set_trainable_state, state_bytes_split, state_num_bytes
from .participation import sample_participation
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


def run_strategy(world: World, strategy: str, out_dir: str, verbose: bool = True) -> dict:
    cfg = world.cfg
    aggregator = get_aggregator(strategy)  # fails early if not implemented
    run_id = make_run_id(cfg, strategy)
    n_clients = cfg["experiment"]["num_clients"]
    if world.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    global_state = {k: v.clone() for k, v in world.init_state.items()}
    cum_up = cum_down = 0

    with RunLogger(os.path.join(out_dir, run_id)) as log:
        log.write_json("config.json", cfg)  # alone enough to regenerate partition and participation
        log.write_json("meta.json", {"run_id": run_id, "strategy": strategy})
        log.write_json("partition_report.json", world.partition.report)

        g, pc = evaluate_global(world, global_state)
        records = [{"run_id": run_id, "strategy": strategy, "round": 0, "skipped": False,
                    "participating_clients": [], "dropped_clients": [], "aggregation_weights": {},
                    "clusters": None, "global_metrics": g, "per_client": pc, "client_train_loss": {},
                    "comm_bytes": {"up": 0, "down": 0, "up_lora": 0, "up_head": 0},
                    "comm_cum_bytes": {"up": 0, "down": 0}, "round_time_s": 0.0}]
        log.log_round(records[0])
        if verbose:
            print(f"[{run_id}] round 0/{cfg['experiment']['rounds']}  acc={g['accuracy']:.3f} f1={g['macro_f1']:.3f}")

        for r in range(1, cfg["experiment"]["rounds"] + 1):
            t0 = time.time()
            participants, dropped = sample_participation(cfg, n_clients, r)
            updates, losses = [], {}
            for cid in participants:
                res = train_client(world.model, global_state, world.enc, world.partition.client_train[cid],
                                   cfg, r, cid, world.device)
                updates.append(ClientUpdate(cid, res.state, res.n_samples))
                losses[str(cid)] = res.mean_loss
            down = state_num_bytes(global_state) * len(participants)
            result = aggregator.aggregate(global_state, updates)
            global_state = result.state
            splits = [state_bytes_split(u.state) for u in updates]
            up_lora, up_head = sum(s[0] for s in splits), sum(s[1] for s in splits)
            up = up_lora + up_head
            cum_up, cum_down = cum_up + up, cum_down + down
            g, pc = evaluate_global(world, global_state)
            rec = {"run_id": run_id, "strategy": strategy, "round": r, "skipped": False,
                   "participating_clients": participants, "dropped_clients": dropped,
                   "aggregation_weights": {str(k): v for k, v in result.weights.items()},
                   "clusters": result.clusters, "global_metrics": g, "per_client": pc,
                   "client_train_loss": losses,
                   "comm_bytes": {"up": up, "down": down, "up_lora": up_lora, "up_head": up_head},
                   "comm_cum_bytes": {"up": cum_up, "down": cum_down},
                   "round_time_s": time.time() - t0}
            records.append(rec)
            log.log_round(rec)
            if verbose:
                print(f"[{run_id}] round {r}/{cfg['experiment']['rounds']}  acc={g['accuracy']:.3f} "
                      f"f1={g['macro_f1']:.3f}  clients={len(participants)}  up={up / 1e6:.2f}MB  "
                      f"{rec['round_time_s']:.1f}s")

        peak = torch.cuda.max_memory_allocated() / 1e6 if world.device.type == "cuda" else None
        summary = build_summary(records, run_id, strategy, time.time() - t_start, peak)
        log.write_json("summary.json", summary)
    return summary


def run_all(cfg: dict, out_dir: str = "runs") -> list:
    """Run every strategy in the config on the SAME partition, seed and starting model (UC5)."""
    strategies = cfg["experiment"]["strategy"]
    for s in strategies:
        get_aggregator(s)  # fail before any slow work
    if cfg["dropout"]["rate"] > 0:
        print("note: dropout is simulated from Step 4; this run uses full participation")
    world = build_world(cfg)
    summaries = []
    for s in strategies:
        set_trainable_state(world.model, world.init_state)
        summaries.append(run_strategy(world, s, out_dir))
    print(f"\n{'strategy':<12} {'final acc':>10} {'final F1':>9} {'worst client':>13} {'upload MB':>10} {'time s':>8}")
    for s in summaries:
        d, c = s["per_client_accuracy_dispersion"], s["cumulative_comm_bytes"]
        print(f"{s['strategy']:<12} {s['final_metrics']['accuracy']:>10.3f} {s['final_metrics']['macro_f1']:>9.3f} "
              f"{d['min']:>13.3f} {c['up'] / 1e6:>10.2f} {s['total_time_s']:>8.1f}")
    print(f"logs written under {out_dir}/")
    return summaries
