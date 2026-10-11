"""Torch-free tests: dropout schedule, network model, deadline, causes, config rules."""
import numpy as np
import pytest

from fedtune.config import ConfigError, load_config
from fedtune.network import TIERS, Profile, client_time, local_steps, make_profiles
from fedtune.participation import ParticipationModel

SIZES = [300, 120, 800, 450, 60, 1000]
PAYLOAD = 3_000_000


def cfg_with(*o):
    return load_config(None, ["experiment.num_clients=6", "experiment.rounds=40"] + list(o))


def model(*o):
    return ParticipationModel(cfg_with(*o), SIZES, PAYLOAD)


# ---- config
def test_network_defaults_off_and_valid():
    c = load_config(None, [])
    assert c["network"]["profile"] == "off" and c["network"]["deadline_s"] is None


def test_deadline_needs_network():
    with pytest.raises(ConfigError) as e:
        load_config(None, ["network.deadline_s=10"])
    assert "network.deadline_s" in str(e.value)


@pytest.mark.parametrize("bad", ["network.slowdown=0.5", "network.jitter=1.0", "network.uplink_mbps=0",
                                 "network.profile=wifi", "network.step_time_s=0"])
def test_bad_network_values(bad):
    with pytest.raises(ConfigError) as e:
        load_config(None, [bad])
    assert bad.split("=")[0] in str(e.value)


# ---- random dropout
def test_rate_zero_and_one():
    m0, m1 = model("dropout.rate=0.0"), model("dropout.rate=1.0")
    assert all(len(m0.select(r).participants) == 6 for r in range(1, 41))
    assert all(m1.select(r).skipped and m1.select(r).participants == [] for r in range(1, 41))


def test_dropout_rate_is_roughly_p():
    m = model("dropout.rate=0.3", "experiment.rounds=200")
    assert abs(m.dropped.mean() - 0.3) < 0.05


def test_schedules_are_nested_across_rates():
    a, b = model("dropout.rate=0.2"), model("dropout.rate=0.5")
    for r in range(1, 41):
        assert set(b.select(r).participants) <= set(a.select(r).participants)


def test_same_seed_same_schedule_and_independent_of_strategy():
    a = model("dropout.rate=0.3").matrix()
    b = model("dropout.rate=0.3", "experiment.strategy=[fedavg, clustered, hybrid]").matrix()
    c = model("dropout.rate=0.3", "experiment.seed=7").matrix()
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_min_participants_skips():
    m = model("dropout.rate=0.6", "dropout.min_participants=4")
    for r in range(1, 41):
        s = m.select(r)
        assert s.skipped == (len(s.participants) < 4)
    assert any(m.select(r).skipped for r in range(1, 41))


def test_dropped_clients_get_no_weight_slot_and_cause():
    s = model("dropout.rate=0.5").select(1)
    assert set(s.dropped) | set(s.selected) == set(range(6)) and not set(s.dropped) & set(s.selected)
    assert all(s.cause[k] == "dropout" for k in s.dropped)
    assert all(s.cause[k] == "participated" for k in s.participants)


# ---- network time model
def test_time_formula_matches_hand_calculation():
    p = Profile("t", up_mbps=10.0, down_mbps=50.0, latency_s=0.05, slowdown=2.0)
    t = client_time(p, 3_000_000, 3_000_000, steps=20, step_time_s=0.1)
    assert t["down"] == pytest.approx(0.05 + 8 * 3e6 / 50e6)
    assert t["compute"] == pytest.approx(0.1 * 2.0 * 20)
    assert t["up"] == pytest.approx(0.05 + 8 * 3e6 / 10e6)
    assert t["total"] == pytest.approx(t["down"] + t["compute"] + t["up"])


def test_halving_uplink_changes_upload_time_by_formula():
    p = Profile("t", 10.0, 50.0, 0.05, 1.0)
    q = Profile("t", 5.0, 50.0, 0.05, 1.0)
    a, b = client_time(p, 3e6, 3e6, 10, 0.1), client_time(q, 3e6, 3e6, 10, 0.1)
    assert b["up"] - a["up"] == pytest.approx(8 * 3e6 / 5e6 - 8 * 3e6 / 10e6)
    assert b["down"] == a["down"] and b["compute"] == a["compute"]


def test_local_steps():
    assert local_steps(33, 16, 1) == 3 and local_steps(32, 16, 2) == 4


def test_tiers_and_assignment():
    c = cfg_with("network.profile=tiered", "network.assignment=large_shards_slow")
    prof = make_profiles(c, SIZES)
    assert prof[int(np.argmax(SIZES))].tier == "slow" and prof[int(np.argmin(SIZES))].tier == "fast"
    c2 = cfg_with("network.profile=tiered", "network.assignment=small_shards_slow")
    prof2 = make_profiles(c2, SIZES)
    assert prof2[int(np.argmin(SIZES))].tier == "slow" and prof2[int(np.argmax(SIZES))].tier == "fast"
    assert {p.tier for p in prof} == {t[0] for t in TIERS}


def test_off_means_no_network_effects():
    s = model("dropout.rate=0.0").select(3)
    assert s.times == {} and s.duration_s == 0.0 and s.late == []


def test_deadline_never_increases_participation():
    prev = None
    for D in (200, 60, 30, 10, 3):
        m = model("network.profile=tiered", f"network.deadline_s={D}", "dropout.rate=0.2")
        n = sum(len(m.select(r).participants) for r in range(1, 41))
        assert prev is None or n <= prev
        prev = n
    assert prev < 6 * 40 * 0.8                     # a 3 s deadline really removes people


def test_late_clients_and_causes_and_duration():
    m = model("network.profile=tiered", "network.deadline_s=30", "dropout.rate=0.0")
    s = m.select(1)
    assert s.late and all(s.times[k]["total"] > 30 for k in s.late)
    assert all(s.times[k]["total"] <= 30 for k in s.participants)
    assert all(s.cause[k] == "deadline" for k in s.late)
    assert s.duration_s == 30.0                      # someone was late: the server waited until the deadline
    m2 = model("network.profile=tiered", "network.deadline_s=1000", "dropout.rate=0.0")
    s2 = m2.select(1)
    assert not s2.late and s2.duration_s == pytest.approx(max(s2.times[k]["total"] for k in range(6)))


def test_dropout_plus_deadline_label_and_skip_duration():
    m = model("network.profile=tiered", "network.deadline_s=20", "dropout.rate=0.5", "dropout.min_participants=6")
    seen = set()
    for r in range(1, 41):
        s = m.select(r)
        seen |= set(s.cause.values())
        assert s.skipped and s.duration_s == 20.0     # all 6 are never on time here
    assert "dropout+deadline" in seen and "deadline" in seen


def test_deadline_decision_is_strategy_independent_and_reproducible():
    a = model("network.profile=tiered", "network.deadline_s=30", "network.jitter=0.3").matrix()
    b = model("network.profile=tiered", "network.deadline_s=30", "network.jitter=0.3",
              "experiment.strategy=[hybrid]").matrix()
    c = model("network.profile=tiered", "network.deadline_s=30", "network.jitter=0.3", "experiment.seed=9").matrix()
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_jitter_changes_times_but_stays_positive():
    m0 = model("network.profile=tiered")
    m1 = model("network.profile=tiered", "network.jitter=0.5")
    t0, t1 = m0.select(2).times, m1.select(2).times
    assert any(abs(t0[k]["up"] - t1[k]["up"]) > 1e-9 for k in range(6))
    assert all(v > 0 for k in range(6) for v in t1[k].values())


def test_matrix_shape_and_consistency():
    m = model("network.profile=tiered", "network.deadline_s=30", "dropout.rate=0.2")
    mat = m.matrix()
    assert mat.shape == (40, 6) and mat.dtype == bool
    assert [int(x) for x in np.flatnonzero(mat[4])] == m.select(5).participants


# ---- YAML reads an unquoted `off` as False; the config loader must still accept it
def test_profile_off_override_and_yaml_file(tmp_path):
    assert load_config(None, ["network.profile=off", "network.deadline_s=null"])["network"]["profile"] == "off"
    f = tmp_path / "c.yaml"
    f.write_text("network: { profile: off }\n")
    assert load_config(str(f))["network"]["profile"] == "off"
    with pytest.raises(ConfigError):
        load_config(None, ["network.profile=true"])
