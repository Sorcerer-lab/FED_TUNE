"""Client summaries from the frozen base encoder (needs torch)."""
from __future__ import annotations

import os

import numpy as np
import torch

from .client_summary import load_summaries, save_summaries, summarise, summary_report
from .data.encoding import make_loader
from .model import set_trainable_state
from .seeding import stream


@torch.no_grad()
def embed_clients(world) -> list:
    """Mean-pooled last-layer hidden states of every client's TRAINING texts, as one array per client.

    The model is put in its initial state, where every LoRA matrix B is zero, so the adapted model
    equals the frozen base model: the embeddings depend only on the pretrained encoder.
    """
    set_trainable_state(world.model, world.init_state)
    world.model.to(world.device)
    world.model.eval()
    shards = world.partition.client_train
    idx = np.concatenate(shards)
    rows = []
    for batch in make_loader(world.enc, idx, world.cfg["train"]["batch_size"] * 2, shuffle=False):
        ids, mask = batch["input_ids"].to(world.device), batch["attention_mask"].to(world.device)
        h = world.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
        m = mask.unsqueeze(-1).to(h.dtype)
        rows.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
    emb = np.concatenate(rows)
    bounds = np.cumsum([0] + [len(s) for s in shards])
    return [emb[bounds[i]:bounds[i + 1]] for i in range(len(shards))]


def ensure_summaries(world, run_dir: str, verbose: bool = True, reuse_file: bool = False) -> dict:
    """Summaries are computed once per experiment and reused by every strategy; a resumed run loads the
    file it saved earlier (reuse_file=True), a fresh run never trusts an old file."""
    path = os.path.join(run_dir, "client_summaries.npz")
    if world.summaries is None:
        if reuse_file and os.path.exists(path):
            world.summaries = load_summaries(path)
        else:
            if verbose:
                print("computing client summaries (frozen-encoder embeddings + k-means) ...")
            K, seed = world.cfg["diversity"]["clusters"], world.cfg["experiment"]["seed"]
            per_client = embed_clients(world)
            world.summaries = {c: summarise(e, K, stream(seed, "summary", c)) for c, e in enumerate(per_client)}
    os.makedirs(run_dir, exist_ok=True)
    save_summaries(path, world.summaries)
    import json
    with open(os.path.join(run_dir, "summary_log.json"), "w", encoding="utf-8") as f:
        json.dump(summary_report(world.summaries), f, indent=2)
    return world.summaries
