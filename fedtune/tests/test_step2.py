import numpy as np
import pytest

from fedtune.config import ConfigError, load_config
from fedtune.data.datasets import _subsample, _synthetic
from fedtune.data.simple_tokenizer import CLS, SEP, SimpleTokenizer
from fedtune.metrics import accuracy, macro_f1
from fedtune.seeding import stream


def cfg_with(*o):
    return load_config(None, list(o))


# ---- config additions
def test_new_keys_have_valid_defaults():
    c = cfg_with()
    assert c["data"]["max_length"] == 128 and c["model"]["lora"]["target_modules"] is None


def test_bad_max_length_names_key():
    with pytest.raises(ConfigError) as e:
        cfg_with("data.max_length=4")
    assert "data.max_length" in str(e.value)


def test_unknown_dataset_rejected():
    with pytest.raises(ConfigError):
        cfg_with("data.dataset=foo")


def test_target_modules_list_accepted():
    assert cfg_with("model.lora.target_modules=[q_lin, v_lin]")["model"]["lora"]["target_modules"] == ["q_lin", "v_lin"]


# ---- seeding with extra keys
def test_stream_keys():
    a = stream(1, "training", 0, 3).random(3)
    assert np.array_equal(a, stream(1, "training", 0, 3).random(3))
    assert not np.array_equal(a, stream(1, "training", 1, 3).random(3))
    assert not np.array_equal(a, stream(1, "training", 0, 4).random(3))


# ---- data
def test_synthetic_deterministic_and_shaped():
    b1, b2 = _synthetic(500, 7), _synthetic(500, 7)
    assert b1.texts == b2.texts and np.array_equal(b1.labels, b2.labels)
    assert len(b1.texts) == 500 and set(np.unique(b1.labels)) == {0, 1, 2, 3}
    assert _synthetic(500, 8).texts != b1.texts


def test_subsample():
    rng = stream(0, "data")
    idx = _subsample(1000, 100, rng)
    assert len(idx) == 100 and len(set(idx)) == 100 and list(idx) == sorted(idx)
    assert len(_subsample(50, 100, stream(0, "data"))) == 50
    assert len(_subsample(50, None, stream(0, "data"))) == 50


def test_simple_tokenizer():
    tok = SimpleTokenizer.fit(["a b c", "a b"])
    out = tok(["a b c"], max_length=32)
    assert out["input_ids"][0][0] == CLS and out["input_ids"][0][-1] == SEP
    assert len(tok(["a b c a b c"], max_length=4)["input_ids"][0]) == 4
    pair = tok(["a"], ["b c"], max_length=32)["input_ids"][0]
    assert pair.count(SEP) == 2
    assert tok(["zzz"])["input_ids"][0][1] == 3  # unknown word


# ---- metrics
def test_macro_f1_known_value():
    y, p = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    # class0: tp1 fp0 fn1 -> 2/3 ; class1: tp2 fp1 fn0 -> 4/5
    assert macro_f1(y, p) == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert accuracy(y, p) == 0.75


def test_macro_f1_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 5, 400), rng.integers(0, 5, 400)
    assert macro_f1(y, p) == pytest.approx(sk.f1_score(y, p, average="macro"))
