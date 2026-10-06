"""Evaluate a model on a set of sample indices (SRS FR-24: accuracy and macro-F1)."""
from __future__ import annotations

import numpy as np
import torch

from .data.encoding import make_loader
from .metrics import accuracy, macro_f1


@torch.no_grad()
def evaluate(model, enc, indices, batch_size: int, device) -> dict:
    model.to(device)
    model.eval()
    preds, loss_sum = [], 0.0
    for batch in make_loader(enc, indices, batch_size, shuffle=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss_sum += float(out.loss.item()) * len(batch["labels"])
        preds.append(out.logits.argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = enc.labels[np.asarray(indices, dtype=int)]
    return {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred),
            "loss": loss_sum / len(y_true), "n": int(len(y_true))}
