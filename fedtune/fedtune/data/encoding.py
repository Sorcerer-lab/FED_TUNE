"""Tokenised data and batching (needs torch)."""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class Encoded:
    input_ids: list      # one list of token ids per sample (NOT padded)
    labels: np.ndarray
    pad_id: int


def encode(bundle, tokenizer, max_length: int) -> Encoded:
    kw = dict(truncation=True, max_length=max_length)
    if bundle.pairs is not None:
        enc = tokenizer(bundle.texts, bundle.pairs, **kw)
    else:
        enc = tokenizer(bundle.texts, **kw)
    pad = tokenizer.pad_token_id
    return Encoded(enc["input_ids"], np.asarray(bundle.labels), 0 if pad is None else int(pad))


class _Rows(Dataset):
    def __init__(self, enc: Encoded, indices):
        self.enc, self.idx = enc, np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return self.enc.input_ids[j], int(self.enc.labels[j])


def _collate(batch, pad_id: int):
    """Pad every sample to the longest one in THIS batch (dynamic padding) and build the
    attention mask: 1 for real tokens, 0 for padding the model must ignore."""
    longest = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for r, (x, _) in enumerate(batch):
        ids[r, : len(x)] = torch.tensor(x, dtype=torch.long)
        mask[r, : len(x)] = 1
    labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def make_loader(enc: Encoded, indices, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(_Rows(enc, indices), batch_size=batch_size, shuffle=shuffle, generator=g,
                      collate_fn=functools.partial(_collate, pad_id=enc.pad_id))
