"""Torch-free tests: config hash, checkpoint bookkeeping, run state, log offsets."""
import json
import os

import pytest

from fedtune.aggregation import get_aggregator
from fedtune.checkpoint import ckpt_path, list_checkpoints, parse_round, prune_rounds, run_state
from fedtune.config import ConfigError, config_hash, load_config
from fedtune.runlog import RunLogger, read_records


def cfg_with(*o):
    return load_config(None, list(o))


def test_checkpoint_config_keys():
    c = cfg_with()
    assert c["checkpoint"] == {"every": 1, "keep": 2}
    for bad in ("checkpoint.every=-1", "checkpoint.keep=0"):
        with pytest.raises(ConfigError) as e:
            cfg_with(bad)
        assert bad.split("=")[0] in str(e.value)


def test_config_hash_rules():
    base = config_hash(cfg_with())
    assert len(base) == 16 and base == config_hash(cfg_with())
    assert config_hash(cfg_with("experiment.rounds=99")) == base            # can extend a run
    assert config_hash(cfg_with("checkpoint.every=5", "checkpoint.keep=3")) == base
    for change in ("train.lr=0.001", "experiment.seed=1", "data.partition.alpha=0.1", "dropout.rate=0.2",
                   "model.lora.rank=4", "network.profile=tiered"):
        assert config_hash(cfg_with(change)) != base, change


def test_names_listing_and_pruning(tmp_path):
    assert parse_round("ckpt_00012.pt") == 12 and parse_round("ckpt_00012.pt.tmp") is None and parse_round("x.pt") is None
    run = str(tmp_path)
    os.makedirs(os.path.dirname(ckpt_path(run, 1)))
    for r in (3, 1, 10):
        open(ckpt_path(run, r), "wb").close()
    open(ckpt_path(run, 11) + ".tmp", "wb").close()              # half-written file must be ignored
    open(os.path.join(run, "checkpoints", "notes.txt"), "w").close()
    assert [r for r, _ in list_checkpoints(run)] == [1, 3, 10]
    assert prune_rounds([1, 3, 10], keep=2) == [1] and prune_rounds([1, 3], keep=2) == []


def test_run_state(tmp_path):
    d = str(tmp_path / "r")
    assert run_state(d) == "none"
    os.makedirs(d)
    assert run_state(d) == "none"                                  # no log yet
    open(os.path.join(d, "rounds.jsonl"), "w").close()
    assert run_state(d) == "unfinished"
    (tmp_path / "r" / "summary.json").write_text(json.dumps({"status": "failed"}))
    assert run_state(d) == "unfinished"
    (tmp_path / "r" / "summary.json").write_text(json.dumps({"status": "completed"}))
    assert run_state(d) == "completed"
    (tmp_path / "r" / "summary.json").write_text("{not json")
    assert run_state(d) == "unfinished"


def test_logger_offset_truncate_and_append(tmp_path):
    d = str(tmp_path / "run")
    with RunLogger(d) as log:
        offsets = []
        for r in range(4):
            log.log_round({"round": r, "text": "caf\u00e9"})          # non-ASCII: bytes != characters
            offsets.append(log.offset())
            assert offsets[-1] == os.path.getsize(os.path.join(d, "rounds.jsonl"))
    path = os.path.join(d, "rounds.jsonl")
    assert [x["round"] for x in read_records(path, offsets[1])] == [0, 1]
    with RunLogger(d, resume_offset=offsets[1]) as log:             # resume after round 1: rounds 2, 3 are dropped
        assert log.offset() == offsets[1]
        log.log_round({"round": 2, "text": "again"})
    recs = read_records(path)
    assert [x["round"] for x in recs] == [0, 1, 2] and recs[2]["text"] == "again"


def test_aggregator_state_hooks_default():
    a = get_aggregator("fedavg")
    assert a.state_dict() == {}
    a.load_state_dict({})
