"""Needs torch + transformers + peft. Skipped automatically when they are not installed."""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch  # noqa: E402

from fedtune.client import train_client  # noqa: E402
from fedtune.config import load_config  # noqa: E402
from fedtune.evaluate import evaluate  # noqa: E402
from fedtune.model import count_params, get_trainable_state, set_trainable_state, state_num_bytes  # noqa: E402
from fedtune.pipeline import build_world  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def world():
    return build_world(load_config("configs/tiny.yaml"), CPU)


def test_only_lora_and_head_are_trainable(world):
    names = list(world.init_state)
    assert any("lora_A" in n for n in names) and any("lora_B" in n for n in names)
    assert any("classifier" in n for n in names)
    assert not any("encoder.layer.0.intermediate" in n for n in names)
    trainable, total = count_params(world.model)
    assert 0 < trainable < total


def test_state_roundtrip(world):
    s = get_trainable_state(world.model)
    set_trainable_state(world.model, s)
    s2 = get_trainable_state(world.model)
    assert all(torch.equal(s[k], s2[k]) for k in s)
    assert state_num_bytes(s) > 0


def test_initial_lora_B_is_zero(world):
    # LoRA starts as a no-op: delta W = B x A = 0 because B is zero-initialised
    assert all(torch.count_nonzero(t) == 0 for n, t in world.init_state.items() if "lora_B" in n)


def test_training_changes_state_and_is_deterministic(world):
    w, cfg = world, world.cfg
    idx = w.partition.client_train[0]
    r1 = train_client(w.model, w.init_state, w.enc, idx, cfg, 0, 0, CPU)
    r2 = train_client(w.model, w.init_state, w.enc, idx, cfg, 0, 0, CPU)
    assert np.isfinite(r1.mean_loss) and r1.steps > 0
    assert any(not torch.equal(r1.state[k], w.init_state[k]) for k in r1.state)
    assert all(torch.equal(r1.state[k], r2.state[k]) for k in r1.state)


def test_global_state_not_mutated_by_client(world):
    before = {k: v.clone() for k, v in world.init_state.items()}
    train_client(world.model, world.init_state, world.enc, world.partition.client_train[1], world.cfg, 0, 1, CPU)
    assert all(torch.equal(before[k], world.init_state[k]) for k in before)


def test_evaluate_ranges(world):
    set_trainable_state(world.model, world.init_state)
    m = evaluate(world.model, world.enc, world.partition.test_idx, 32, CPU)
    assert 0 <= m["accuracy"] <= 1 and 0 <= m["macro_f1"] <= 1 and m["n"] == len(world.partition.test_idx)


def test_training_beats_chance_on_synthetic(world):
    w = world
    res = train_client(w.model, w.init_state, w.enc, np.concatenate(w.partition.client_train), w.cfg, 0, 0, CPU)
    set_trainable_state(w.model, res.state)
    m = evaluate(w.model, w.enc, w.partition.test_idx, 32, CPU)
    assert m["accuracy"] > 0.35  # chance is 0.25 with 4 classes
