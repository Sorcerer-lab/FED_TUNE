"""Base model + LoRA adapter, and helpers to read/write ONLY the trainable parameters.

The trainable parameters (LoRA matrices + the classification head) are exactly what a client
uploads and what the server sends back. The big frozen base model is never transmitted.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from peft import LoraConfig, TaskType, get_peft_model

from .seeding import stream

TINY_ID = "tiny-random"  # offline random-weight BERT used for the pipeline check
AUTO_TARGETS = {
    "distilbert": ["q_lin", "v_lin"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "albert": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
}


def build_model(cfg: dict, num_labels: int, vocab_size: int | None = None):
    base = cfg["model"]["base"]
    seed = cfg["experiment"]["seed"]
    # one seed fixes the initial classifier head and LoRA matrices: the global starting point
    torch.manual_seed(int(stream(seed, "model_init").integers(0, 2**31 - 1)))
    if base == TINY_ID:
        from transformers import BertConfig, BertForSequenceClassification
        mc = BertConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64,
                        max_position_embeddings=256, num_labels=num_labels)
        model = BertForSequenceClassification(mc)
    else:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=num_labels)

    lora = cfg["model"]["lora"]
    targets = lora["target_modules"] or AUTO_TARGETS.get(model.config.model_type)
    if not targets:
        raise ValueError(f"unknown model type '{model.config.model_type}': set model.lora.target_modules")
    lcfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora["rank"], lora_alpha=lora["alpha"],
                      lora_dropout=lora["dropout"], target_modules=targets)
    return get_peft_model(model, lcfg)


def get_trainable_state(model) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((n, p.detach().cpu().clone()) for n, p in model.named_parameters() if p.requires_grad)


def set_trainable_state(model, state) -> None:
    params = dict(model.named_parameters())
    missing = [n for n in state if n not in params]
    if missing:
        raise KeyError(f"state has parameters the model does not: {missing[:3]}")
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t.to(device=params[n].device, dtype=params[n].dtype))


def state_num_bytes(state) -> int:
    return int(sum(t.numel() * t.element_size() for t in state.values()))


def count_params(model) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def state_bytes_split(state) -> tuple[int, int]:
    """(bytes in LoRA matrices, bytes in everything else i.e. the classification head)."""
    lora = sum(t.numel() * t.element_size() for n, t in state.items() if "lora_" in n)
    return int(lora), state_num_bytes(state) - int(lora)
