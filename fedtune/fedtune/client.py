"""Local training of one simulated client (SRS FR-9)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .data.encoding import make_loader
from .model import get_trainable_state, set_trainable_state
from .seeding import stream


@dataclass
class TrainResult:
    state: dict          # the client's updated trainable parameters (what it would upload)
    n_samples: int
    mean_loss: float
    steps: int


def train_client(model, global_state, enc, indices, cfg, round_idx: int, client_id: int, device) -> TrainResult:
    """Start from the global trainable state, fine-tune on this client's shard, return the result.
    One shared base model is reused for every client; only the trainable state is swapped in."""
    t = cfg["train"]
    seed = int(stream(cfg["experiment"]["seed"], "training", round_idx, client_id).integers(0, 2**31 - 1))
    torch.manual_seed(seed)  # fixes LoRA dropout masks

    set_trainable_state(model, global_state)
    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=0.0)  # fresh optimiser every round
    loader = make_loader(enc, indices, t["batch_size"], shuffle=True, seed=seed)

    total, steps = 0.0, 0
    for _ in range(t["local_epochs"]):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += float(loss.item())
            steps += 1
    return TrainResult(get_trainable_state(model), len(indices), total / max(steps, 1), steps)
