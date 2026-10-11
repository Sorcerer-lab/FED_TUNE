"""Who takes part in each round (SRS UC4, UC12): random dropout + network model + deadline.

The whole schedule is a pure function of the configuration, so it is identical for every
aggregation strategy (fair comparison) and for every resumed run. Random dropout uses one matrix
of uniform numbers drawn once; client k misses round r when u[r,k] < p. Because the numbers are
fixed by the seed, schedules for two dropout rates p1 < p2 are nested: everyone absent at p1 is
also absent at p2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .network import client_time, local_steps, make_profiles
from .seeding import stream


@dataclass
class RoundParticipation:
    round: int
    selected: list            # clients that were available (not randomly dropped)
    participants: list        # selected and on time
    dropped: list             # randomly unavailable
    late: list                # selected but missed the deadline
    cause: dict               # client -> "participated" | "dropout" | "deadline" | "dropout+deadline"
    times: dict               # client -> {"down","compute","up","total"} (empty when network is off)
    skipped: bool             # fewer than min_participants on time
    duration_s: float         # simulated duration of the round


class ParticipationModel:
    def __init__(self, cfg: dict, shard_sizes, payload_bytes: int):
        self.cfg = cfg
        self.R = cfg["experiment"]["rounds"]
        self.K = len(shard_sizes)
        seed = cfg["experiment"]["seed"]
        self.p = float(cfg["dropout"]["rate"])
        self.m_min = cfg["dropout"]["min_participants"]
        net = cfg["network"]
        self.deadline = net["deadline_s"]
        self.dropped = stream(seed, "dropout").random((self.R, self.K)) < self.p
        self.profiles = make_profiles(cfg, shard_sizes)
        self.payload = int(payload_bytes)
        self.steps = [local_steps(n, cfg["train"]["batch_size"], cfg["train"]["local_epochs"]) for n in shard_sizes]
        self.tau = float(net["step_time_s"])
        self.jitter = float(net["jitter"])
        self._u = stream(seed, "network", 1).random((self.R, self.K, 2)) if self.profiles else None

    def _times(self, r: int) -> dict:
        if self.profiles is None:
            return {}
        out = {}
        for k in range(self.K):
            fu = 1.0 + self.jitter * (2 * self._u[r, k, 0] - 1)
            fd = 1.0 + self.jitter * (2 * self._u[r, k, 1] - 1)
            out[k] = client_time(self.profiles[k], self.payload, self.payload, self.steps[k], self.tau, fu, fd)
        return out

    def select(self, round_idx: int) -> RoundParticipation:
        """round_idx is 1-based."""
        i = round_idx - 1
        times = self._times(i)
        dropped = [k for k in range(self.K) if self.dropped[i, k]]
        selected = [k for k in range(self.K) if not self.dropped[i, k]]
        is_late = {k: bool(self.deadline is not None and times and times[k]["total"] > self.deadline) for k in range(self.K)}
        late = [k for k in selected if is_late[k]]
        participants = [k for k in selected if not is_late[k]]
        cause = {}
        for k in range(self.K):
            if k in dropped:
                cause[k] = "dropout+deadline" if is_late[k] else "dropout"
            else:
                cause[k] = "deadline" if is_late[k] else "participated"
        skipped = len(participants) < self.m_min
        if self.profiles is None:
            dur = 0.0
        elif self.deadline is not None and (skipped or late):
            dur = float(self.deadline)
        else:
            dur = max([times[k]["total"] for k in participants], default=0.0)
        return RoundParticipation(round_idx, selected, participants, dropped, late, cause, times, skipped, dur)

    def matrix(self) -> np.ndarray:
        """R x K boolean matrix: True where the client was an on-time participant."""
        m = np.zeros((self.R, self.K), dtype=bool)
        for r in range(1, self.R + 1):
            m[r - 1, self.select(r).participants] = True
        return m
