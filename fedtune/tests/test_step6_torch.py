"""Needs torch + peft + transformers. Summaries from the frozen encoder and diversity-aware runs, end to end."""
import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

import torch  # noqa: E402

from fedtune.config import load_config  # noqa: E402
from fedtune.embedding import embed_clients, ensure_summaries  # noqa: E402
from fedtune.pipeline import build_world  # noqa: E402
from fedtune.runlog import make_run_id, read_records  # noqa: E402
from fedtune.runner import run_strategy  # noqa: E402

CPU = torch.device("cpu")
BASE = ["network.profile=off", "network.deadline_s=null", "dropout.rate=0.0", "train.local_epochs=1"]


def world(*o):
    return build_world(load_config("configs/step6_tiny.yaml", BASE + list(o)), CPU)


def rundir(w, out, strategy):
    return os.path.join(str(out), make_run_id(w.cfg, strategy))


def recs(d):
    out = read_records(os.path.join(d, "rounds.jsonl"))
    for r in out:
        r.pop("round_time_s", None)
    return out


def same(a, b, tol=1e-4):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    return abs(a - b) <= tol if isinstance(a, float) else a == b


def test_embeddings_have_right_shape_and_are_deterministic():
    w = world()
    e1, e2 = embed_clients(w), embed_clients(w)
    assert len(e1) == w.cfg["experiment"]["num_clients"]
    for k, e in enumerate(e1):
        assert e.shape == (len(w.partition.client_train[k]), 32) and np.isfinite(e).all()
        assert np.allclose(e, e2[k], atol=1e-6)


def test_summaries_are_bounded_and_logged(tmp_path):
    w = world()
    sums = ensure_summaries(w, str(tmp_path), verbose=False)
    K = w.cfg["diversity"]["clusters"]
    for c, s in sums.items():
        assert s.centroids.shape[0] <= K and s.counts.sum() == len(w.partition.client_train[c])
        assert np.allclose(np.linalg.norm(s.centroids, axis=1), 1.0, atol=1e-4)
    log = json.load(open(tmp_path / "summary_log.json"))
    assert log["total_bytes"] == sum(s.nbytes for s in sums.values()) and "no text" in log["reveals"]
    assert (tmp_path / "client_summaries.npz").exists()


def test_diversity_run_logs_gains_weights_and_summary_bytes(tmp_path):
    w = world("experiment.rounds=3")
    s = run_strategy(w, "diversity", str(tmp_path), verbose=False)
    r = recs(rundir(w, tmp_path, "diversity"))
    total = json.load(open(os.path.join(rundir(w, tmp_path, "diversity"), "summary_log.json")))["total_bytes"]
    assert r[0]["comm_cum_bytes"]["up"] == total > 0 and r[0]["comm_bytes"]["up_summary"] == total
    for x in r[1:]:
        assert sum(x["aggregation_weights"].values()) == pytest.approx(1.0)
        assert set(x["aggregator_info"]["coverage_gain"]) == set(x["aggregation_weights"])
    assert s["status"] == "completed"
    f = run_strategy(w, "fedavg", str(tmp_path), verbose=False)
    assert recs(rundir(w, tmp_path, "fedavg"))[0]["comm_cum_bytes"]["up"] == 0 and f["status"] == "completed"


def test_gamma_zero_reproduces_fedavg(tmp_path):
    w = world("experiment.rounds=3", "diversity.gamma=0")
    run_strategy(w, "fedavg", str(tmp_path / "a"), verbose=False)
    run_strategy(w, "diversity", str(tmp_path / "b"), verbose=False)
    ra, rb = recs(rundir(w, tmp_path / "a", "fedavg")), recs(rundir(w, tmp_path / "b", "diversity"))
    for x, y in zip(ra[1:], rb[1:]):
        assert same(x["aggregation_weights"], y["aggregation_weights"], 1e-9)
        assert same(x["global_metrics"], y["global_metrics"])


def test_resume_of_diversity_run_equals_uninterrupted(tmp_path):
    def crash(rnd, cid, state):
        if rnd == 4:
            raise RuntimeError("disconnect")
        return state
    cfg = ["experiment.rounds=5", "checkpoint.every=2"]
    a, b = world(*cfg), world(*cfg)
    run_strategy(a, "diversity", str(tmp_path / "A"), verbose=False)
    with pytest.raises(RuntimeError):
        run_strategy(b, "diversity", str(tmp_path / "B"), verbose=False, fault_hook=crash)
    fresh = world(*cfg)                                    # new process: summaries must come from the saved file
    assert fresh.summaries is None
    run_strategy(fresh, "diversity", str(tmp_path / "B"), verbose=False, resume=True)
    assert fresh.summaries is not None
    assert same(recs(rundir(a, tmp_path / "A", "diversity")), recs(rundir(b, tmp_path / "B", "diversity")))
