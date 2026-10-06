import numpy as np
import pytest

from fedtune.config import ConfigError, load_config
from fedtune.data.partition import build_partition
from fedtune.seeding import stream


def make_data(n=6000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, n), rng.integers(0, 4, n)


def cfg_with(*overrides):
    return load_config(None, list(overrides))


# ---- configuration (UC1 / FR-1 / FR-2)
def test_defaults_valid():
    assert load_config("configs/smoke.yaml")["experiment"]["num_clients"] == 6


def test_override_applies():
    assert cfg_with("dropout.rate=0.35")["dropout"]["rate"] == 0.35


def test_bad_dropout_names_key():
    with pytest.raises(ConfigError) as e:
        cfg_with("dropout.rate=1.4")
    assert "dropout.rate" in str(e.value) and "[0, 1]" in str(e.value)


def test_unknown_key_rejected():
    with pytest.raises(ConfigError) as e:
        cfg_with("dropout.rat=0.1")
    assert "dropout.rat" in str(e.value)


def test_min_participants_cannot_exceed_clients():
    with pytest.raises(ConfigError):
        cfg_with("experiment.num_clients=4", "dropout.min_participants=5")


def test_reference_config_valid():
    assert load_config("configs/reference.yaml")["grid"]["seeds"] == [1, 2, 3]


# ---- reproducibility (FR-3)
def test_streams_independent_and_reproducible():
    a1 = stream(7, "partition").random(5)
    a2 = stream(7, "partition").random(5)
    b = stream(7, "dropout").random(5)
    assert np.array_equal(a1, a2) and not np.array_equal(a1, b)


@pytest.mark.parametrize("mode", ["dirichlet", "quantity", "domain"])
def test_same_seed_same_partition(mode):
    labels, groups = make_data()
    c = cfg_with(f"data.partition.mode={mode}")
    p1, p2 = build_partition(labels, c, groups), build_partition(labels, c, groups)
    assert all(np.array_equal(a, b) for a, b in zip(p1.client_train, p2.client_train))
    c2 = cfg_with(f"data.partition.mode={mode}", "experiment.seed=43")
    p3 = build_partition(labels, c2, groups)
    assert not all(np.array_equal(a, b) for a, b in zip(p1.client_train, p3.client_train))


# ---- correctness (FR-5 / FR-7)
@pytest.mark.parametrize("mode", ["dirichlet", "quantity", "domain"])
def test_disjoint_and_complete(mode):
    labels, groups = make_data()
    p = build_partition(labels, cfg_with(f"data.partition.mode={mode}"), groups)
    allidx = np.concatenate([p.test_idx, *p.client_train, *p.client_val])
    assert len(allidx) == len(labels) and len(np.unique(allidx)) == len(labels)


def test_every_client_has_validation_data():
    labels, _ = make_data()
    p = build_partition(labels, cfg_with())
    assert all(len(v) >= 1 for v in p.client_val)


def test_lower_alpha_means_higher_skew():
    labels, _ = make_data()
    lo = build_partition(labels, cfg_with("data.partition.alpha=0.1")).report["label_heterogeneity_js"]
    hi = build_partition(labels, cfg_with("data.partition.alpha=5.0")).report["label_heterogeneity_js"]
    assert lo > hi


def test_domain_heterogeneity_monotone():
    labels, groups = make_data()
    scores = []
    for h in (0.0, 0.5, 1.0):
        c = cfg_with("data.partition.mode=domain", f"data.partition.heterogeneity={h}")
        scores.append(build_partition(labels, c, groups).report["group_heterogeneity_js"])
    assert scores[0] < scores[1] < scores[2]


def test_quantity_skew_changes_sizes_not_labels():
    labels, _ = make_data()
    r = build_partition(labels, cfg_with("data.partition.mode=quantity", "data.partition.alpha=0.5")).report
    assert r["size_max"] > 2 * r["size_min"]
    assert r["label_heterogeneity_js"] < 0.05


def test_report_has_no_raw_text_fields():
    labels, groups = make_data()
    r = build_partition(labels, cfg_with("data.partition.mode=domain"), groups).report
    assert set(r["clients"][0]) <= {"client", "n_train", "n_val", "label_counts", "group_counts"}
