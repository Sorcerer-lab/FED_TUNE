# Step 4: Dropout, the network model, deadlines, skipped rounds and failures

**Design coverage:** random dropout and minimum participants, the analytic network model (bandwidth, latency, compute speed, round deadline), simulated time, handling of broken updates, crash handling.
**Goal of this step:** make "who takes part in a round" realistic and **identical for every aggregation strategy**, and make the run survive bad updates and crashes. Aggregation is still FedAvg (new strategies come in Steps 6 to 8).

> **Testing note:** the dropout schedule, network model, deadline logic and configuration rules are tested here (65 tests pass here, including 25 new ones). The round loop needs PyTorch, which my sandbox cannot run, so `tests/test_step4_torch.py` is for you to run. Paste any error and I'll fix it.

## 1. Four reasons a client can be missing from a round
| Cause (label in the log) | Where it is decided | What happens |
|---|---|---|
| `dropout` | random draw per (round, client) | the client is unavailable: it gets no model, trains nothing |
| `deadline` | simulated time of the client exceeds the deadline | the client is too slow ("straggler"): its update is ignored |
| `dropout+deadline` | both would apply | recorded, so the two effects can be separated |
| `non_finite` | after local training | the update contains NaN or Inf and is excluded |

Everyone else is a **participant**. If fewer than `dropout.min_participants` remain, the round is **skipped**.

## 2. Random dropout (UC4)
- **One matrix, drawn once:** for all rounds R and clients K we draw a matrix of uniform random numbers `u[r, k]` from its own random stream (`dropout`). Client k misses round r when `u[r, k] < p`.
- **Why a pre-drawn matrix:** the schedule then depends only on the seed, never on what a strategy does with random numbers, so FedAvg, diversity-aware and clustered aggregation all see the **same** absences (fair comparison).
- **Nested schedules (useful for the heatmap):** because the numbers are fixed, everyone absent at p = 0.2 is also absent at p = 0.5. Going along the dropout axis of your grid therefore only *adds* absences instead of reshuffling them, which reduces noise. A test checks this.
- **Skipped round:** if `|participants| < min_participants`, nobody trains, the global state is unchanged, and the log repeats the previous metrics with `skipped: true` and a `skip_reason`. Downloads to the available clients are still counted as spent bytes.

## 3. The network model (UC12)
Every client gets a **profile**: uplink and downlink **bandwidth** (Mbit/s, i.e. millions of bits per second), one-way **latency** (the fixed delay per transfer), and a **compute slowdown** (1 = reference speed, 3 = three times slower).
- `homogeneous`: everyone gets the values from the config.
- `tiered`: clients are split into thirds, **fast / medium / slow**:

| Tier | Uplink | Downlink | Latency | Compute |
|---|---|---|---|---|
| fast | 50 Mbit/s | 100 Mbit/s | 20 ms | ×1.0 |
| medium | 10 Mbit/s | 50 Mbit/s | 60 ms | ×1.5 |
| slow | 1 Mbit/s | 10 Mbit/s | 150 ms | ×3.0 |

**Simulated time of one client in one round:**
```
t = [latency + 8·bytes_down / rate_down]  +  [step_time · slowdown · steps]  +  [latency + 8·bytes_up / rate_up]
      download                                local compute                         upload
```
(the factor 8 turns bytes into bits; `steps = ceil(shard size / batch size) × local epochs`; `step_time` is the reference seconds per training step.)

**Worked example** (DistilBERT update 2.964 MB, step time 0.1 s, a slow client with 747 samples, so 47 steps): download 0.15 + 8·2.964e6/10e6 = 2.5 s; compute 0.1 · 3 · 47 = 14.1 s; upload 0.15 + 8·2.964e6/1e6 = 23.9 s; total ≈ **40.5 s**. A fast client with 507 samples needs about 4 s. The upload over a slow uplink dominates, which is exactly the networking lesson of the project.

**Jitter** (`network.jitter`, default 0) multiplies each bandwidth by a random factor in [1−j, 1+j], drawn per (round, client) from the `network` stream. A client that is right at the deadline then succeeds in some rounds and fails in others.

**Assignment of tiers** (`network.assignment`): `random` (seeded), `large_shards_slow` (clients with the most data get the slowest links), or `small_shards_slow`. The correlated options matter: with a deadline, a slow client with a big shard is *always* late, so its data **never reaches the model**. That is a bias that random dropout cannot cause. The demo below shows it.

## 4. Deadline and simulated round duration
- `network.deadline_s = D` (or `null` = wait for everyone). A selected client with `t > D` is a straggler.
- **Round duration:** if any selected client is late, or the round is skipped, the server waited until D, so the duration is **D**. Otherwise it is the slowest participant's `t`. Without a network model the duration is 0.
- **Assumption (stated, not hidden):** a randomly dropped client is *unavailable at selection time*, so the server does not wait for it; only deadline misses stretch a round.
- Late clients are **not trained** in the simulation: their update would be thrown away, and this saves GPU time. Their download is still counted.
- The log accumulates **simulated time** (`sim_time_cum_s`), so later you can plot accuracy against simulated seconds, not only against rounds.

## 5. Why the outcome never depends on measured time or on the strategy
Participation is a pure function of (seed, configuration, update size, shard sizes). The step time is a configured constant, not a stopwatch, and the update size used is the strategy-independent trainable-state size (later, the small client summaries are counted in communication cost but never decide who is late). Separate random streams (`dropout`, `network`) keep all draws independent of the partition and training. The default step time 0.1 s comes from your Step 3 run (about 38 s per round for 6 clients with about 67 steps each, evaluation included, so it is a slightly generous upper bound); you can recalibrate it.

## 6. Broken updates (UC8)
After a client trains, its update is checked: if any tensor has NaN or Inf (a too-large learning rate or numeric overflow can cause this), the update is **excluded**, logged as `{"client": k, "reason": "non_finite"}`, and the run continues with the rest. If too few valid updates remain, the round is skipped with `skip_reason: too_few_valid_updates`. FedAvg's weights are always renormalised over the valid updates.

## 7. Crashes
Any exception inside a run is caught, a `summary.json` with `"status": "failed"` and the error text is written, the log file is closed with every finished round intact, and the error is re-raised. (Checkpoint and resume come in Step 5.)

## 8. What is new in the logs
Per round: `selected_clients`, `dropped_clients`, `late_clients`, `excluded_clients`, `cause` (per client), `client_times` (down / compute / up / total per client), `skip_reason`, `sim_time_s`, `sim_time_cum_s`. Per run: `participation.npy` (rounds × clients, True = on-time participant) and in `summary.json`: `status`, `skip_reasons`, `absences` (counts per cause) and `simulated_time_s`.

## 9. New configuration keys (`network` section)
| Key | Default | Allowed |
|---|---|---|
| `profile` | `off` | `off`, `homogeneous`, `tiered` |
| `uplink_mbps`, `downlink_mbps` | 10, 50 | > 0 (used by `homogeneous`) |
| `latency_ms` | 50 | ≥ 0 |
| `slowdown` | 1.0 | ≥ 1 |
| `deadline_s` | `null` | `null` or > 0 (needs profile ≠ `off`) |
| `step_time_s` | 0.1 | > 0 |
| `jitter` | 0.0 | [0, 1) |
| `assignment` | `random` | `random`, `large_shards_slow`, `small_shards_slow` |

Note: in YAML files and on the command line, an unquoted `off` is read as the boolean False; the loader converts it back to the word `off`, so `profile: off` and `--set network.profile=off` both work.

## 10. How to run
**Laptop, no training and no torch (seconds):** explore the schedule and the times.
```
python -m fedtune network-demo configs/step4_real.yaml
python -m fedtune network-demo configs/step4_real.yaml --set network.deadline_s=80
python -m fedtune network-demo configs/step4_real.yaml --set network.assignment=large_shards_slow
python -m fedtune network-demo configs/step4_real.yaml --set dropout.rate=0.5 --set network.deadline_s=null --set network.profile=off
```
(The demo uses synthetic labels, so the shard sizes are close to, but not exactly, those of a real AG News run.)

**Laptop, full pipeline on the tiny model:**
```
python -m pytest -q
python -m fedtune run configs/step4_tiny.yaml --out runs
```
**Colab (GPU), real model:**
```
%cd fedtune
!pip -q install peft datasets
!pip uninstall -y torchao
!python -m pytest -q
!python -m fedtune run configs/step4_real.yaml --out runs
```

## 11. What you should observe
- In the demo output, client 3 (the largest shard, slow tier, about 61 s) is **late in every round**; client 1 is right at the deadline and flips between rounds. This is the bias described in section 3.
- `round` lines print `ok / drop / late / bad` counts and the simulated seconds; a deadline of 40 s makes most rounds last exactly 40 s.
- On Colab, accuracy should still rise, but more slowly and noisily than in Step 3, because fewer clients take part each round and some data are rarely seen.
- Experiments worth running (and writing about): deadline 20 / 40 / 80 s; `assignment=random` versus `large_shards_slow`; dropout 0 / 0.2 / 0.5. Compare accuracy **per round** and **per simulated second**: a shorter deadline gives faster rounds but loses the slow clients' data.

## 12. Things to understand for your viva
- Why draw the whole dropout matrix once, and what does "nested" mean?
- What exactly is in `t`, and why does the upload dominate for slow clients?
- Why must participation not depend on measured time or on the strategy?
- Why can a deadline bias the model while random dropout does not?
- What is the difference between a skipped round and a round with fewer participants?

## 13. Limitations
The network model is analytic (no congestion, loss or protocol overhead); dropout is independent per round and happens before the round (no mid-round failures); one deadline for all rounds; bytes of rejected or late uploads are not counted as waste; the step time is a constant.

## Next: Step 5
Checkpoint and resume: save the global state, round counter and log position after every round, so a disconnected Colab session continues exactly where it stopped, with the same participation.
