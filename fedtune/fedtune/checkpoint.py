"""Atomic checkpoints and run-state detection (SRS UC9, FR 27, 34).

A checkpoint is small: the global trainable state, the round number, the byte offset of rounds.jsonl,
a fingerprint of the configuration, strategy state and elapsed time. Nothing else is needed because
every random number in FedTune is derived from (seed, round, client) rather than from a generator
that is advanced over time: the partition, the participation schedule and each client's batch order
can be recomputed identically after a restart. The frozen base model is rebuilt from its checkpoint.
"""
from __future__ import annotations

import json
import os
import re

CKPT_DIR = "checkpoints"
VERSION = 1


class CheckpointError(RuntimeError):
    pass


def ckpt_path(run_dir: str, round_idx: int) -> str:
    return os.path.join(run_dir, CKPT_DIR, f"ckpt_{round_idx:05d}.pt")


def parse_round(name: str):
    m = re.fullmatch(r"ckpt_(\d{5})\.pt", name)
    return int(m.group(1)) if m else None


def list_checkpoints(run_dir: str) -> list:
    """[(round, path)] sorted by round; half-written *.tmp files are ignored."""
    d = os.path.join(run_dir, CKPT_DIR)
    if not os.path.isdir(d):
        return []
    found = [(parse_round(n), os.path.join(d, n)) for n in os.listdir(d)]
    return sorted((r, p) for r, p in found if r is not None)


def prune_rounds(rounds: list, keep: int) -> list:
    """Which of the (sorted) checkpoint rounds to delete so that only the newest `keep` remain."""
    return rounds[:-keep] if len(rounds) > keep else []


def run_state(run_dir: str) -> str:
    """'none' (nothing there), 'completed' (finished normally) or 'unfinished' (crashed, failed or killed)."""
    if not os.path.exists(os.path.join(run_dir, "rounds.jsonl")):
        return "none"
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        try:
            with open(sp, encoding="utf-8") as f:
                if json.load(f).get("status") == "completed":
                    return "completed"
        except (OSError, ValueError):
            pass
    return "unfinished"


def save_checkpoint(run_dir, round_idx, global_state, strategy_state, log_offset, cfg_hash, elapsed_s, keep=2) -> str:
    """Write to a temporary file, fsync, then rename: a crash can never leave a half-written checkpoint."""
    import torch
    os.makedirs(os.path.join(run_dir, CKPT_DIR), exist_ok=True)
    final = ckpt_path(run_dir, round_idx)
    tmp = final + ".tmp"
    payload = {"version": VERSION, "round": int(round_idx), "global_state": dict(global_state),
               "strategy_state": strategy_state, "log_offset": int(log_offset),
               "config_hash": cfg_hash, "elapsed_s": float(elapsed_s)}
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    rounds = [r for r, _ in list_checkpoints(run_dir)]
    for r in prune_rounds(rounds, keep):
        os.remove(ckpt_path(run_dir, r))
    return final


def load_latest(run_dir: str, cfg_hash: str | None = None):
    """Newest readable checkpoint, or None. An unreadable newest file falls back to the previous one.
    A checkpoint from a different configuration is refused."""
    import torch
    for r, p in reversed(list_checkpoints(run_dir)):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if ck.get("version") != VERSION:
            continue
        if cfg_hash is not None and ck["config_hash"] != cfg_hash:
            raise CheckpointError(
                f"checkpoint {os.path.basename(p)} was made with a different configuration "
                f"(hash {ck['config_hash']} vs {cfg_hash}); use --overwrite to start this run over")
        return ck
    return None
