"""Torch-free tests: aggregation maths, logging and summary."""
import json

import numpy as np
import pytest

from fedtune.aggregation import StrategyNotImplemented, get_aggregator
from fedtune.aggregation.base import ClientUpdate, normalise, weighted_average
from fedtune.config import load_config
from fedtune.metrics import dispersion
from fedtune.runlog import RunLogger, build_summary, make_run_id


def upd(cid, n, a, b):
    return ClientUpdate(cid, {"lora_A": np.array(a, float), "head": np.array(b, float)}, n)


def test_weighted_average_known_values():
    s1, s2 = {"x": np.array([0.0, 4.0])}, {"x": np.array([4.0, 0.0])}
    out = weighted_average([s1, s2], [0.25, 0.75])
    assert np.allclose(out["x"], [3.0, 1.0])


def test_identical_states_average_to_themselves():
    s = {"x": np.array([1.0, 2.0])}
    assert np.allclose(weighted_average([s, s, s], [0.2, 0.3, 0.5])["x"], s["x"])


def test_mismatched_keys_rejected():
    with pytest.raises(ValueError):
        weighted_average([{"a": np.zeros(1)}, {"b": np.zeros(1)}], [0.5, 0.5])


def test_normalise_rules():
    assert normalise([1, 3]).tolist() == [0.25, 0.75]
    for bad in ([0, 0], [-1, 2], [float("nan"), 1], []):
        with pytest.raises(ValueError):
            normalise(bad)


def test_fedavg_weights_follow_data_size():
    agg = get_aggregator("fedavg")
    res = agg.aggregate({}, [upd(0, 100, [0, 0], [0]), upd(3, 300, [4, 4], [8])])
    assert res.weights == {0: 0.25, 3: 0.75}
    assert sum(res.weights.values()) == pytest.approx(1.0)
    assert np.allclose(res.state["lora_A"], [3, 3]) and np.allclose(res.state["head"], [6])


def test_fedavg_over_subset_renormalises():
    res = get_aggregator("fedavg").aggregate({}, [upd(1, 50, [1], [1]), upd(2, 50, [3], [3])])
    assert res.weights == {1: 0.5, 2: 0.5}


def test_zero_updates_rejected():
    with pytest.raises(ValueError):
        get_aggregator("fedavg").aggregate({}, [])


def test_planned_and_unknown_strategies():
    with pytest.raises(StrategyNotImplemented) as e:
        get_aggregator("diversity")
    assert "Step 6" in str(e.value)
    with pytest.raises(StrategyNotImplemented):
        get_aggregator("nonsense")


def test_dispersion():
    d = dispersion([0.5, 0.7, 0.9])
    assert d["min"] == 0.5 and d["max"] == 0.9 and d["mean"] == pytest.approx(0.7)


def test_run_id():
    c = load_config(None, ["data.partition.alpha=0.1", "dropout.rate=0.2", "experiment.seed=3"])
    assert make_run_id(c, "fedavg") == "ag_news-dirichlet0p1-drop0p2-fedavg-s3"


def _rec(r, acc, parts):
    return {"round": r, "skipped": False, "participating_clients": parts,
            "global_metrics": {"accuracy": acc, "macro_f1": acc, "loss": 1.0},
            "per_client": {"0": {"accuracy": acc - 0.1}, "1": {"accuracy": acc + 0.1}},
            "comm_cum_bytes": {"up": 10 * r, "down": 20 * r}}


def test_summary():
    recs = [_rec(0, 0.2, []), _rec(1, 0.5, [0, 1]), _rec(2, 0.7, [0])]
    s = build_summary(recs, "id", "fedavg", 12.5, None)
    assert s["rounds_completed"] == 2 and s["rounds_skipped"] == 0
    assert s["final_metrics"]["accuracy"] == 0.7 and s["initial_metrics"]["accuracy"] == 0.2
    assert s["mean_participants_per_round"] == 1.5
    assert s["per_client_accuracy_dispersion"]["min"] == pytest.approx(0.6)
    assert s["cumulative_comm_bytes"] == {"up": 20, "down": 40}


def test_logger_keeps_lines_after_crash(tmp_path):
    with pytest.raises(RuntimeError):
        with RunLogger(str(tmp_path / "run")) as log:
            log.log_round({"round": 0, "v": np.float32(1.5)})
            log.log_round({"round": 1, "v": np.int64(2)})
            raise RuntimeError("boom")
    lines = (tmp_path / "run" / "rounds.jsonl").read_text().strip().splitlines()
    assert [json.loads(x)["round"] for x in lines] == [0, 1]
