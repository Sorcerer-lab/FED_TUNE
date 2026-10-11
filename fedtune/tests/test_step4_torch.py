"""Needs torch + peft + transformers. End-to-end checks of dropout, deadline, skipped rounds and faults."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch  # noqa: E402

from fedtune.config import load_config  # noqa: E402
from fedtune.model import state_num_bytes  # noqa: E402
from fedtune.participation import ParticipationModel  # noqa: E402
from fedtune.pipeline import build_world  # noqa: E402
from fedtune.runner import run_strategy  # noqa: E402

CPU = torch.device("cpu")
NO_NET = ["network.profile=off", "network.deadline_s=null"]


def world(*o):
    return build_world(load_config("configs/step4_tiny.yaml", ["experiment.rounds=4", "train.local_epochs=1"] + list(o)), CPU)


def records(tmp_path, summary):
    return [json.loads(x) for x in (tmp_path / summary["run_id"] / "rounds.jsonl").read_text().splitlines()]


def test_dropout_bytes_participation_and_weights(tmp_path):
    w = world(*NO_NET, "dropout.rate=0.5", "dropout.min_participants=2", "experiment.rounds=6")
    s = run_strategy(w, "fedavg", str(tmp_path), verbose=False)
    recs = records(tmp_path, s)
    pm = ParticipationModel(w.cfg, [len(x) for x in w.partition.client_train], state_num_bytes(w.init_state))
    payload = state_num_bytes(w.init_state)
    prev = recs[0]["global_metrics"]
    for r in recs[1:]:
        sel = pm.select(r["round"])
        assert r["dropped_clients"] == sel.dropped and r["selected_clients"] == sel.selected
        assert not set(r["dropped_clients"]) & set(r["participating_clients"])
        assert r["comm_bytes"]["down"] == payload * len(sel.selected)
        if r["skipped"]:
            assert r["participating_clients"] == [] and r["comm_bytes"]["up"] == 0
            assert r["global_metrics"] == prev                 # global model unchanged
        else:
            assert r["participating_clients"] == sel.participants
            assert r["comm_bytes"]["up"] == payload * len(sel.participants)
            assert sum(r["aggregation_weights"].values()) == pytest.approx(1.0)
        prev = r["global_metrics"]
    assert s["status"] == "completed" and s["absences"]["dropout"] == sum(len(r["dropped_clients"]) for r in recs[1:])
    assert np.array_equal(np.load(tmp_path / s["run_id"] / "participation.npy"), pm.matrix())


def test_everyone_dropped_skips_every_round(tmp_path):
    w = world(*NO_NET, "dropout.rate=1.0")
    s = run_strategy(w, "fedavg", str(tmp_path), verbose=False)
    assert s["rounds_skipped"] == 4 and s["rounds_completed"] == 0
    assert s["skip_reasons"] == {"too_few_participants": 4}
    assert s["final_metrics"] == s["initial_metrics"]


def test_late_clients_are_excluded_and_not_trained(tmp_path):
    w = world("dropout.rate=0.0", "network.deadline_s=2.0", "network.assignment=large_shards_slow")
    s = run_strategy(w, "fedavg", str(tmp_path), verbose=False)
    recs = records(tmp_path, s)[1:]
    assert any(r["late_clients"] for r in recs)
    for r in recs:
        assert not set(r["late_clients"]) & set(r["participating_clients"])
        assert {int(k) for k in r["client_train_loss"]} <= set(r["participating_clients"]) | {e["client"] for e in r["excluded_clients"]}
        assert all(r["cause"][str(k)] == "deadline" for k in r["late_clients"])
    assert s["simulated_time_s"] > 0 and s["absences"]["deadline"] > 0


def test_non_finite_update_is_excluded_and_logged(tmp_path):
    def hook(rnd, cid, state):
        if rnd == 1 and cid == 0:
            k = next(iter(state)); state[k] = state[k] * float("nan")
        return state
    w = world(*NO_NET, "dropout.rate=0.0", "experiment.rounds=2")
    s = run_strategy(w, "fedavg", str(tmp_path), verbose=False, fault_hook=hook)
    r1 = records(tmp_path, s)[1]
    assert r1["excluded_clients"] == [{"client": 0, "reason": "non_finite"}] and r1["cause"]["0"] == "non_finite"
    assert 0 not in r1["participating_clients"] and sum(r1["aggregation_weights"].values()) == pytest.approx(1.0)
    assert np.isfinite(r1["global_metrics"]["loss"]) and s["absences"]["non_finite"] == 1


def test_too_few_valid_updates_skips_round(tmp_path):
    def hook(rnd, cid, state):
        if rnd == 1 and cid == 0:
            k = next(iter(state)); state[k] = state[k] * float("inf")
        return state
    n = 6
    w = world(*NO_NET, "dropout.rate=0.0", f"dropout.min_participants={n}", "experiment.rounds=2")
    s = run_strategy(w, "fedavg", str(tmp_path), verbose=False, fault_hook=hook)
    recs = records(tmp_path, s)
    assert recs[1]["skipped"] and recs[1]["skip_reason"] == "too_few_valid_updates"
    assert recs[1]["global_metrics"] == recs[0]["global_metrics"] and not recs[2]["skipped"]


def test_crash_keeps_logs_and_writes_failed_status(tmp_path):
    def hook(rnd, cid, state):
        if rnd == 2:
            raise RuntimeError("boom")
        return state
    w = world(*NO_NET, "dropout.rate=0.0")
    with pytest.raises(RuntimeError):
        run_strategy(w, "fedavg", str(tmp_path), verbose=False, fault_hook=hook)
    from fedtune.runlog import make_run_id
    d = tmp_path / make_run_id(w.cfg, "fedavg")
    assert json.loads((d / "summary.json").read_text())["status"] == "failed"
    assert [json.loads(x)["round"] for x in (d / "rounds.jsonl").read_text().splitlines()] == [0, 1]
