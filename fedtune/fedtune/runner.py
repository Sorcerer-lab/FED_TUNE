"""The federated round loop (SRS UC3, UC4, UC5, UC8, UC12)."""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import torch

from .aggregation import get_aggregator
from .aggregation.base import AggContext, ClientUpdate
from .checkpoint import CKPT_DIR, load_latest, run_state, save_checkpoint
from .client import train_client
from .config import config_hash
from .embedding import ensure_summaries
from .evaluate import evaluate
from .model import is_finite_state, set_trainable_state, state_bytes_split, state_num_bytes
from .participation import ParticipationModel
from .pipeline import World, build_world
from .runlog import RunLogger, build_summary, make_run_id, read_records

_KEEP = ("accuracy", "macro_f1", "loss")


class RunExistsError(RuntimeError):
    pass


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


def run_strategy(world: World, strategy: str, out_dir: str, verbose: bool = True, fault_hook=None,
                 resume: bool = False, overwrite: bool = False) -> dict:
    """fault_hook(round, client_id, state) -> state is for tests: it can corrupt or crash an update.
    resume=True continues an unfinished run from its latest checkpoint (or starts fresh if there is none)
    and returns the stored summary of a completed run; an unfinished run is never overwritten silently."""
    cfg = world.cfg
    aggregator = get_aggregator(strategy, cfg)  # fails early if not implemented
    run_id = make_run_id(cfg, strategy)
    run_dir = os.path.join(out_dir, run_id)
    rounds, m_min = cfg["experiment"]["rounds"], cfg["dropout"]["min_participants"]
    every, keep = cfg["checkpoint"]["every"], cfg["checkpoint"]["keep"]
    chash = config_hash(cfg)
    state, ck = run_state(run_dir), None
    if resume:
        if state == "completed":
            with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
                done = json.load(f)
            if done["rounds_completed"] + done["rounds_skipped"] >= rounds:
                if verbose:
                    print(f"[{run_id}] already completed, nothing to do")
                return done
        ck = load_latest(run_dir, chash)           # raises CheckpointError for a different configuration
        if ck is None and state != "none" and verbose:
            print(f"[{run_id}] no usable checkpoint found, starting from scratch")
    elif state == "unfinished" and not overwrite:
        raise RunExistsError(f"{run_dir} holds an unfinished run: use --resume to continue it "
                             f"or --overwrite to start over")
    if ck is None:
        shutil.rmtree(os.path.join(run_dir, CKPT_DIR), ignore_errors=True)   # never mix old checkpoints in
    if world.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_session = time.time()
    payload = state_num_bytes(world.init_state)
    pm = ParticipationModel(cfg, [len(s) for s in world.partition.client_train], payload)
    summaries = ensure_summaries(world, run_dir, verbose, reuse_file=ck is not None) if aggregator.needs_summary else None
    summary_bytes = sum(x.nbytes for x in summaries.values()) if summaries else 0
    ctx = AggContext(summaries=summaries, n_clients=len(world.partition.client_train))
    if ck is not None:
        r0, elapsed_prev = ck["round"], ck["elapsed_s"]
        global_state = {k: v.clone() for k, v in ck["global_state"].items()}
        aggregator.load_state_dict(ck["strategy_state"])
        records = read_records(os.path.join(run_dir, "rounds.jsonl"), ck["log_offset"])
        assert records and records[-1]["round"] == r0, "log and checkpoint disagree"
        logger = RunLogger(run_dir, resume_offset=ck["log_offset"])
        last = records[-1]
        g, pc = last["global_metrics"], last["per_client"]
        cum_up, cum_down = last["comm_cum_bytes"]["up"], last["comm_cum_bytes"]["down"]
        sim_cum = last.get("sim_time_cum_s", 0.0)
        if verbose:
            print(f"[{run_id}] resuming after round {r0} (checkpoint), continuing to round {rounds}")
    else:
        r0, elapsed_prev, records = 0, 0.0, []
        global_state = {k: v.clone() for k, v in world.init_state.items()}
        logger = RunLogger(run_dir)
        cum_up, cum_down = summary_bytes, 0       # summaries are uploaded once, at setup
        sim_cum = 0.0

    def checkpoint(r):
        if every > 0:
            save_checkpoint(run_dir, r, global_state, aggregator.state_dict(), log.offset(), chash,
                            elapsed_prev + time.time() - t_session, keep)

    with logger as log:
        log.write_json("config.json", cfg)  # alone enough to regenerate partition and participation
        log.write_json("meta.json", {"run_id": run_id, "strategy": strategy})
        log.write_json("partition_report.json", world.partition.report)
        np.save(os.path.join(log.dir, "participation.npy"), pm.matrix())
        try:
            if ck is None:
                g, pc = evaluate_global(world, global_state)
                records.append({"run_id": run_id, "strategy": strategy, "round": 0, "skipped": False,
                                "skip_reason": None, "selected_clients": [], "participating_clients": [],
                                "dropped_clients": [], "late_clients": [], "excluded_clients": [], "cause": {},
                                "client_times": {}, "aggregation_weights": {}, "clusters": None,
                                "global_metrics": g, "per_client": pc, "client_train_loss": {},
                                "comm_bytes": {"up": summary_bytes, "down": 0, "up_lora": 0, "up_head": 0,
                                               "up_summary": summary_bytes},
                                "comm_cum_bytes": {"up": summary_bytes, "down": 0}, "round_time_s": 0.0,
                                "sim_time_s": 0.0, "sim_time_cum_s": 0.0})
                log.log_round(records[0])
                checkpoint(0)
                if verbose:
                    print(f"[{run_id}] round 0/{rounds}  acc={g['accuracy']:.3f} f1={g['macro_f1']:.3f}")

            for r in range(r0 + 1, rounds + 1):
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
                weights, clusters, ainfo = {}, None, None
                if not skipped:
                    result = aggregator.aggregate(global_state, updates, ctx)
                    global_state = result.state
                    weights = {str(k): v for k, v in result.weights.items()}
                    clusters, ainfo = result.clusters, result.info
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
                       "aggregation_weights": weights, "clusters": clusters, "aggregator_info": ainfo,
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
                if every > 0 and (r % every == 0 or r == rounds):
                    checkpoint(r)
        except BaseException as e:  # keep logs readable and record the failure (SRS UC8, FR-34)
            log.write_json("summary.json", {"run_id": run_id, "strategy": strategy, "status": "failed",
                                            "error": f"{type(e).__name__}: {e}", "rounds_logged": len(records) - 1})
            raise
        peak = torch.cuda.max_memory_allocated() / 1e6 if world.device.type == "cuda" else None
        summary = build_summary(records, run_id, strategy, elapsed_prev + time.time() - t_session, peak)
        log.write_json("summary.json", summary)
    return summary


def run_all(cfg: dict, out_dir: str = "runs", resume: bool = False, overwrite: bool = False) -> list:
    """Run every strategy in the config on the SAME partition, seed, participation and starting model (UC5)."""
    strategies = cfg["experiment"]["strategy"]
    for s in strategies:
        get_aggregator(s, cfg)  # fail before any slow work
    world = build_world(cfg)
    summaries = []
    for s in strategies:
        set_trainable_state(world.model, world.init_state)
        summaries.append(run_strategy(world, s, out_dir, resume=resume, overwrite=overwrite))
    print(f"\n{'strategy':<12} {'final acc':>10} {'final F1':>9} {'worst client':>13} {'skipped':>8} {'upload MB':>10} {'sim s':>8} {'time s':>8}")
    for s in summaries:
        d, c = s["per_client_accuracy_dispersion"], s["cumulative_comm_bytes"]
        print(f"{s['strategy']:<12} {s['final_metrics']['accuracy']:>10.3f} {s['final_metrics']['macro_f1']:>9.3f} "
              f"{d['min']:>13.3f} {s.get('rounds_skipped', 0):>8} {c['up'] / 1e6:>10.2f} {s.get('simulated_time_s', 0.0):>8.1f} {s['total_time_s']:>8.1f}")
    print(f"logs written under {out_dir}/")
    return summaries
