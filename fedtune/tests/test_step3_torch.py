"""Needs torch + peft + transformers. End-to-end check of the round loop on the tiny offline config."""
import json
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch  # noqa: E402

from fedtune.config import load_config  # noqa: E402
from fedtune.model import state_num_bytes  # noqa: E402
from fedtune.pipeline import build_world  # noqa: E402
from fedtune.runner import run_strategy  # noqa: E402

CPU = torch.device("cpu")


def _world():
    cfg = load_config("configs/tiny.yaml", ["experiment.rounds=2", "train.local_epochs=1"])
    return build_world(cfg, CPU)


def test_two_round_run_writes_expected_files(tmp_path):
    w = _world()
    s = run_strategy(w, "fedavg", str(tmp_path), verbose=False)
    run_dir = tmp_path / s["run_id"]
    assert {"rounds.jsonl", "summary.json", "config.json", "meta.json", "partition_report.json"} <= set(os.listdir(run_dir))
    recs = [json.loads(x) for x in (run_dir / "rounds.jsonl").read_text().splitlines()]
    assert [r["round"] for r in recs] == [0, 1, 2]
    r1 = recs[1]
    n = w.cfg["experiment"]["num_clients"]
    assert len(r1["participating_clients"]) == n
    assert sum(r1["aggregation_weights"].values()) == pytest.approx(1.0)
    per_state = state_num_bytes(w.init_state)
    assert r1["comm_bytes"]["up"] == n * per_state and r1["comm_bytes"]["down"] == n * per_state
    assert r1["comm_bytes"]["up_lora"] + r1["comm_bytes"]["up_head"] == r1["comm_bytes"]["up"]
    assert recs[2]["comm_cum_bytes"]["up"] == 2 * n * per_state
    assert 0 <= r1["global_metrics"]["accuracy"] <= 1
    assert s["rounds_completed"] == 2


def test_same_seed_same_metrics(tmp_path):
    a = run_strategy(_world(), "fedavg", str(tmp_path / "a"), verbose=False)
    b = run_strategy(_world(), "fedavg", str(tmp_path / "b"), verbose=False)
    assert a["final_metrics"]["accuracy"] == pytest.approx(b["final_metrics"]["accuracy"], abs=1e-4)
    assert a["final_metrics"]["loss"] == pytest.approx(b["final_metrics"]["loss"], abs=1e-4)
