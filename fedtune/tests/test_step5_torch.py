"""Needs torch + peft + transformers. Checkpoint and resume, end to end."""
import json
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch  # noqa: E402

from fedtune.checkpoint import CheckpointError, list_checkpoints, load_latest, save_checkpoint  # noqa: E402
from fedtune.config import config_hash, load_config  # noqa: E402
from fedtune.pipeline import build_world  # noqa: E402
from fedtune.runlog import make_run_id, read_records  # noqa: E402
from fedtune.runner import RunExistsError, run_strategy  # noqa: E402

CPU = torch.device("cpu")
BASE = ["network.profile=off", "network.deadline_s=null", "dropout.rate=0.0", "train.local_epochs=1"]


def world(*o):
    return build_world(load_config("configs/step4_tiny.yaml", BASE + list(o)), CPU)


def same(a, b, tol=1e-4):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float):
        return abs(a - b) <= tol
    return a == b


def recs(d):
    out = read_records(os.path.join(d, "rounds.jsonl"))
    for r in out:
        r.pop("round_time_s", None)
    return out


def rundir(w, out):
    return os.path.join(str(out), make_run_id(w.cfg, "fedavg"))


def crash_at(n):
    def hook(rnd, cid, state):
        if rnd == n:
            raise RuntimeError("simulated disconnect")
        return state
    return hook


def test_roundtrip_atomic_and_pruning(tmp_path):
    state = {"a": torch.arange(4.0), "b": torch.ones(2, 2)}
    for r in (1, 2, 3, 4):
        save_checkpoint(str(tmp_path), r, state, {}, 100 * r, "h", 1.5, keep=2)
    assert [r for r, _ in list_checkpoints(str(tmp_path))] == [3, 4]
    assert not [n for n in os.listdir(tmp_path / "checkpoints") if n.endswith(".tmp")]
    ck = load_latest(str(tmp_path), "h")
    assert ck["round"] == 4 and ck["log_offset"] == 400 and ck["elapsed_s"] == 1.5
    assert all(torch.equal(ck["global_state"][k], state[k]) for k in state)


def test_corrupt_newest_falls_back_and_hash_mismatch_refused(tmp_path):
    state = {"a": torch.zeros(3)}
    for r in (1, 2):
        save_checkpoint(str(tmp_path), r, state, {}, r, "h", 0.0, keep=2)
    with open(list_checkpoints(str(tmp_path))[-1][1], "wb") as f:
        f.write(b"garbage")
    assert load_latest(str(tmp_path), "h")["round"] == 1
    with pytest.raises(CheckpointError):
        load_latest(str(tmp_path), "other-config")


def test_resume_after_crash_equals_uninterrupted_run(tmp_path):
    a, b = world("experiment.rounds=5", "checkpoint.every=2"), world("experiment.rounds=5", "checkpoint.every=2")
    run_strategy(a, "fedavg", str(tmp_path / "A"), verbose=False)
    with pytest.raises(RuntimeError):
        run_strategy(b, "fedavg", str(tmp_path / "B"), verbose=False, fault_hook=crash_at(4))
    db = rundir(b, tmp_path / "B")
    assert [r["round"] for r in recs(db)] == [0, 1, 2, 3]            # round 3 was logged but not checkpointed
    assert [r for r, _ in list_checkpoints(db)] == [0, 2]
    fresh = world("experiment.rounds=5", "checkpoint.every=2")        # a new process rebuilds everything
    s = run_strategy(fresh, "fedavg", str(tmp_path / "B"), verbose=False, resume=True)
    rb, ra = recs(db), recs(rundir(a, tmp_path / "A"))
    assert [r["round"] for r in rb] == [0, 1, 2, 3, 4, 5]             # no duplicate or missing rounds
    assert same(ra, rb)                                                # identical metrics, weights, bytes, participation
    assert s["status"] == "completed" and s["rounds_completed"] == 5


def test_completed_run_resume_is_a_no_op(tmp_path):
    w = world("experiment.rounds=2")
    s1 = run_strategy(w, "fedavg", str(tmp_path), verbose=False)
    before = open(os.path.join(rundir(w, tmp_path), "rounds.jsonl"), "rb").read()
    s2 = run_strategy(w, "fedavg", str(tmp_path), verbose=False, resume=True)
    assert s2["run_id"] == s1["run_id"] and open(os.path.join(rundir(w, tmp_path), "rounds.jsonl"), "rb").read() == before


def test_unfinished_run_is_protected_until_overwrite(tmp_path):
    w = world("experiment.rounds=3")
    with pytest.raises(RuntimeError):
        run_strategy(w, "fedavg", str(tmp_path), verbose=False, fault_hook=crash_at(2))
    with pytest.raises(RunExistsError):
        run_strategy(w, "fedavg", str(tmp_path), verbose=False)
    s = run_strategy(w, "fedavg", str(tmp_path), verbose=False, overwrite=True)
    assert s["status"] == "completed" and [r["round"] for r in recs(rundir(w, tmp_path))] == [0, 1, 2, 3]


def test_changed_configuration_cannot_resume(tmp_path):
    w = world("experiment.rounds=3")
    with pytest.raises(RuntimeError):
        run_strategy(w, "fedavg", str(tmp_path), verbose=False, fault_hook=crash_at(2))
    other = world("experiment.rounds=3", "train.lr=0.001")
    with pytest.raises(CheckpointError):
        run_strategy(other, "fedavg", str(tmp_path), verbose=False, resume=True)
    assert config_hash(w.cfg) != config_hash(other.cfg)


def test_finished_run_can_be_extended_with_more_rounds(tmp_path):
    w3 = world("experiment.rounds=3")
    run_strategy(w3, "fedavg", str(tmp_path), verbose=False)
    first = recs(rundir(w3, tmp_path))
    w5 = world("experiment.rounds=5")
    s = run_strategy(w5, "fedavg", str(tmp_path), verbose=False, resume=True)
    out = recs(rundir(w5, tmp_path))
    assert [r["round"] for r in out] == [0, 1, 2, 3, 4, 5] and same(first, out[:4]) and s["rounds_completed"] == 5


def test_resume_without_checkpoints_starts_fresh(tmp_path):
    w = world("experiment.rounds=2", "checkpoint.every=0")
    with pytest.raises(RuntimeError):
        run_strategy(w, "fedavg", str(tmp_path), verbose=False, fault_hook=crash_at(2))
    s = run_strategy(w, "fedavg", str(tmp_path), verbose=False, resume=True)
    assert s["status"] == "completed" and not list_checkpoints(rundir(w, tmp_path))
