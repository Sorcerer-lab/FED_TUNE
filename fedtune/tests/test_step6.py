"""Torch-free tests: summaries, coverage gains and diversity-aware weights."""
import numpy as np
import pytest

from fedtune.aggregation import get_aggregator
from fedtune.aggregation.base import AggContext, ClientUpdate
from fedtune.aggregation.diversity import DiversityAware, cap_weights, coverage_gains
from fedtune.client_summary import ClientSummary, kmeans, load_summaries, save_summaries, summarise, summary_report
from fedtune.config import ConfigError, load_config
from fedtune.seeding import stream


def cfg_with(*o):
    return load_config(None, list(o))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def summ(points, counts=None):
    c = np.array([unit(p) for p in points], dtype=np.float32)
    return ClientSummary(c, np.array(counts if counts is not None else [10] * len(c), dtype=np.int32))


def updates(ns):
    return [ClientUpdate(i, {}, n) for i, n in enumerate(ns)]


# ---- k-means and summaries
def blobs(rng, centers, per=60, noise=0.05):
    return np.concatenate([np.array(c) + noise * rng.normal(size=(per, len(c))) for c in centers])


def test_kmeans_recovers_separated_blobs_and_is_deterministic():
    X = blobs(np.random.default_rng(0), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    C1, a1 = kmeans(X, 3, stream(1, "summary", 0))
    C2, a2 = kmeans(X, 3, stream(1, "summary", 0))
    assert np.array_equal(C1, C2) and np.array_equal(a1, a2)
    assert sorted(np.bincount(a1)) == [60, 60, 60]


def test_summarise_shape_bytes_and_unit_norm():
    X = np.random.default_rng(1).normal(size=(200, 32))
    s = summarise(X, 8, stream(1, "summary", 3))
    assert s.centroids.shape == (8, 32) and s.counts.sum() == 200
    assert np.allclose(np.linalg.norm(s.centroids, axis=1), 1.0, atol=1e-5)
    assert s.nbytes == 8 * 32 * 4 + 8 * 4                      # K*d floats + K int32 counts: bounded size


def test_summarise_few_samples_and_empty_cells():
    s = summarise(np.random.default_rng(0).normal(size=(3, 5)), 8, stream(1, "summary", 0))
    assert len(s.counts) <= 3 and s.counts.sum() == 3 and (s.counts > 0).all()


def test_save_load_roundtrip_and_report(tmp_path):
    d = {0: summ([[1, 0], [0, 1]], [5, 7]), 2: summ([[1, 1]], [4])}
    save_summaries(str(tmp_path / "s.npz"), d)
    back = load_summaries(str(tmp_path / "s.npz"))
    assert sorted(back) == [0, 2] and np.array_equal(back[0].counts, [5, 7])
    assert np.allclose(back[2].centroids, d[2].centroids)
    rep = summary_report(d)
    assert rep["total_bytes"] == sum(x.nbytes for x in d.values()) and "no text" in rep["reveals"]


# ---- coverage gains
def test_redundant_client_has_zero_gain_unique_client_positive():
    sums = {0: summ([[1, 0]]), 1: summ([[1, 0]]), 2: summ([[0, 1]])}
    g = coverage_gains(sums, [0, 1, 2], [0, 1, 2])
    assert g[0] == pytest.approx(0, abs=1e-9) and g[1] == pytest.approx(0, abs=1e-9)
    assert g[2] > 0.5


def test_gain_of_single_client_is_all_its_coverage():
    sums = {0: summ([[1, 0], [0, 1]], [3, 4])}
    assert coverage_gains(sums, [0], [0])[0] == pytest.approx(7.0)


def test_universe_all_rewards_clients_covering_absent_ones():
    # clients 0 and 1 are present; client 2 (absent) holds data like client 0
    sums = {0: summ([[1, 0]]), 1: summ([[0, 1]]), 2: summ([[1, 0.1]])}
    part = coverage_gains(sums, [0, 1], [0, 1])
    allu = coverage_gains(sums, [0, 1], [0, 1, 2])
    assert allu[0] > part[0] and allu[1] == pytest.approx(part[1], abs=0.2)


# ---- weights
def test_weights_valid_and_unique_client_gains_weight():
    sums = {0: summ([[1, 0]]), 1: summ([[1, 0]]), 2: summ([[0, 1]])}
    agg = DiversityAware(cfg_with())
    w = agg.weights(updates([100, 100, 100]), AggContext(sums, 3))
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    assert w[2] > w[0] and w[0] == pytest.approx(w[1])


def test_gamma_zero_is_exactly_fedavg():
    sums = {0: summ([[1, 0]]), 1: summ([[0, 1]]), 2: summ([[1, 1]])}
    ns = [50, 150, 300]
    w = DiversityAware(cfg_with("diversity.gamma=0")).weights(updates(ns), AggContext(sums, 3))
    assert np.allclose(w, np.array(ns) / sum(ns))
    w2 = get_aggregator("fedavg").weights(updates(ns))
    assert np.allclose(w, w2)


def test_beta_zero_ignores_data_volume():
    sums = {0: summ([[1, 0]]), 1: summ([[0, 1]]), 2: summ([[1, 1]])}
    cfg = cfg_with("diversity.beta=0", "diversity.gain=total")      # (per_sample divides the gain by n by design)
    a = DiversityAware(cfg).weights(updates([10, 10, 10]), AggContext(sums, 3))
    b = DiversityAware(cfg).weights(updates([10, 1000, 99999]), AggContext(sums, 3))
    assert np.allclose(a, b)


def test_cap_respected_and_sums_to_one():
    w = cap_weights(np.array([0.7, 0.2, 0.1]), 0.5)
    assert w.max() <= 0.5 + 1e-9 and w.sum() == pytest.approx(1.0) and w[0] > w[1] > w[2]
    assert cap_weights(np.array([0.5, 0.5]), 0.1).tolist() == [0.5, 0.5]            # infeasible cap falls back
    assert cap_weights(np.array([0.7, 0.3]), None).tolist() == [0.7, 0.3]
    sums = {0: summ([[1, 0]]), 1: summ([[1, 0]]), 2: summ([[0, 1]])}
    w = DiversityAware(cfg_with("diversity.max_weight=0.4", "diversity.epsilon=0.001")).weights(updates([10, 10, 10]), AggContext(sums, 3))
    assert w.max() <= 0.4 + 1e-9 and w.sum() == pytest.approx(1.0)


def test_single_participant_gets_weight_one_and_missing_summaries_fail():
    sums = {0: summ([[1, 0]])}
    assert DiversityAware(cfg_with()).weights(updates([5]), AggContext(sums, 1)).tolist() == [1.0]
    with pytest.raises(ValueError):
        DiversityAware(cfg_with()).weights(updates([5, 6]), None)


def test_info_is_logged_and_aggregate_uses_context():
    agg = get_aggregator("diversity", cfg_with())
    assert agg.needs_summary and not get_aggregator("fedavg").needs_summary
    sums = {0: summ([[1, 0]]), 1: summ([[0, 1]])}
    ups = [ClientUpdate(0, {"x": np.zeros(2)}, 10), ClientUpdate(1, {"x": np.ones(2)}, 10)]
    res = agg.aggregate({}, ups, AggContext(sums, 2))
    assert sum(res.weights.values()) == pytest.approx(1.0) and set(res.info["coverage_gain"]) == {"0", "1"}


def test_config_validation():
    for bad in ("diversity.clusters=0", "diversity.gamma=-1", "diversity.epsilon=0", "diversity.max_weight=1.5",
                "diversity.universe=everyone"):
        with pytest.raises(ConfigError) as e:
            cfg_with(bad)
        assert bad.split("=")[0] in str(e.value)
    assert cfg_with("diversity.max_weight=0.5")["diversity"]["max_weight"] == 0.5


def test_per_sample_gain_does_not_count_volume_twice():
    # two clients with completely different data; client 1 has 4x the samples, so 4x the raw coverage mass
    sums = {0: summ([[1, 0]], [100]), 1: summ([[0, 1]], [400])}
    ns = [100, 400]
    per = DiversityAware(cfg_with("diversity.gain=per_sample")).weights(updates(ns), AggContext(sums, 2))
    tot = DiversityAware(cfg_with("diversity.gain=total")).weights(updates(ns), AggContext(sums, 2))
    assert per[1] / per[0] == pytest.approx(4.0, rel=0.01)      # volume counted once, as in FedAvg
    assert tot[1] / tot[0] == pytest.approx(16.0, rel=0.01)     # the old total gain counted it twice


def test_gain_option_validation_and_logging():
    with pytest.raises(ConfigError):
        cfg_with("diversity.gain=weird")
    agg = DiversityAware(cfg_with())
    sums = {0: summ([[1, 0]]), 1: summ([[0, 1]])}
    agg.weights(updates([10, 30]), AggContext(sums, 2))
    assert set(agg.info()) >= {"coverage_gain", "gain_raw", "universe", "gain"}
