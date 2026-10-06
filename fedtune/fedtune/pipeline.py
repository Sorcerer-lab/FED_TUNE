"""Builds everything one experiment needs ("the world") and runs the Step-2 local check."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .client import train_client
from .data.datasets import Bundle, load_bundle
from .data.encoding import Encoded, encode
from .data.partition import Partition, build_partition
from .data.simple_tokenizer import SimpleTokenizer
from .evaluate import evaluate
from .model import TINY_ID, build_model, count_params, get_trainable_state, set_trainable_state, state_num_bytes


class PipelineError(RuntimeError):
    pass


@dataclass
class World:
    cfg: dict
    bundle: Bundle
    enc: Encoded
    partition: Partition
    model: object
    init_state: dict
    num_labels: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_world(cfg: dict, device=None) -> World:
    mode, name, base = cfg["data"]["partition"]["mode"], cfg["data"]["dataset"], cfg["model"]["base"]
    allowed = {"domain": ("synthetic", "multi_nli"), "language": ("synthetic", "xnli")}
    if mode in allowed and name not in allowed[mode]:
        raise PipelineError(f"partition mode '{mode}' needs dataset in {list(allowed[mode])}, got '{name}'")
    if name == "xnli" and base != TINY_ID and not any(s in base.lower() for s in ("multilingual", "xlm", "mdeberta")):
        print(f"warning: xnli is multilingual but model.base='{base}' looks English-only")
    if name == "synthetic" and base != TINY_ID:
        print("note: synthetic data with a real pretrained model is only useful for debugging")

    bundle = load_bundle(cfg)
    if base == TINY_ID:
        tokenizer = SimpleTokenizer.fit(bundle.texts + (bundle.pairs or []))
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base)
    enc = encode(bundle, tokenizer, cfg["data"]["max_length"])
    partition = build_partition(bundle.labels, cfg, bundle.groups)
    num_labels = len(bundle.label_names)
    model = build_model(cfg, num_labels, vocab_size=getattr(tokenizer, "vocab_size", None))
    return World(cfg, bundle, enc, partition, model, get_trainable_state(model), num_labels,
                 device or pick_device())


def run_local_check(cfg: dict, max_clients: int | None = None) -> None:
    """Train each client once from the SAME starting point and evaluate. No merging yet (Step 3)."""
    w = build_world(cfg)
    bs = cfg["train"]["batch_size"]
    trainable, total = count_params(w.model)
    up = state_num_bytes(w.init_state)
    full = total * 4
    print(f"device={w.device}  samples={len(w.bundle.labels)}  classes={w.num_labels}")
    print(f"trainable parameters {trainable:,} of {total:,} ({100 * trainable / total:.3f}%)")
    print(f"one upload = {up / 1e6:.3f} MB = {100 * up / full:.3f}% of the full model "
          f"({full / 1e6:.1f} MB at 32-bit)  [SRS NFR-3 target: <= 5%]")
    r = w.partition.report
    print(f"partition: mode={r['mode']} label JS={r['label_heterogeneity_js']:.3f}")

    set_trainable_state(w.model, w.init_state)
    base = evaluate(w.model, w.enc, w.partition.test_idx, bs, w.device)
    print(f"before training: global-test accuracy={base['accuracy']:.3f} macro-F1={base['macro_f1']:.3f}")
    print(f"{'client':>6} {'n_train':>8} {'steps':>6} {'loss':>7} {'own-val acc':>12} {'global-test acc':>16} {'sec':>6}")
    n = len(w.partition.client_train) if max_clients is None else min(max_clients, len(w.partition.client_train))
    for cid in range(n):
        t0 = time.time()
        res = train_client(w.model, w.init_state, w.enc, w.partition.client_train[cid], cfg, 0, cid, w.device)
        own = evaluate(w.model, w.enc, w.partition.client_val[cid], bs, w.device)
        glob = evaluate(w.model, w.enc, w.partition.test_idx, bs, w.device)
        print(f"{cid:>6} {res.n_samples:>8} {res.steps:>6} {res.mean_loss:>7.3f} "
              f"{own['accuracy']:>12.3f} {glob['accuracy']:>16.3f} {time.time() - t0:>6.1f}")
