# Step 5: Checkpoint and resume

**Design coverage:** UC9 (checkpoint and resume), crash safety of logs and checkpoints, session resilience on free GPUs.
**Goal:** if a run is interrupted (Colab disconnects, you close the laptop, Ctrl+C, a crash), you restart with one flag and it continues after the last saved round, producing the **same results** as an uninterrupted run, with no duplicated or missing rounds in the log.

> **Testing note:** the bookkeeping (configuration fingerprint, checkpoint file naming and pruning, run-state detection, log offsets and truncation) is tested here (71 tests pass). Writing and loading real checkpoints and the resume-equals-uninterrupted test need PyTorch, which my sandbox cannot run, so `tests/test_step5_torch.py` is for you to run. Paste any error and I'll fix it.

## 1. The problem
A full grid is hours of GPU time on a session that can vanish at any moment. Without checkpoints, a disconnect after round 17 of 20 throws away everything.

## 2. What must be saved, and why it is so little
A **checkpoint** is a snapshot of a run that is enough to continue it. Ours holds only:
- the **global trainable state** (LoRA matrices plus head, about 3 MB for DistilBERT);
- the **round number**;
- the **byte offset** of the log file (section 4);
- a **fingerprint** of the configuration (section 5);
- the **strategy state** (empty for FedAvg; later, things like per-client matrices for FedSA-LoRA);
- the elapsed time.

Nothing about random generators is stored, **by design**: every random number in FedTune is *derived* from (seed, round, client), not drawn from a generator that advances over time. The partition, the dropout matrix, the network jitter and each client's batch order can therefore be recomputed identically after a restart. The frozen base model is simply rebuilt from the model hub. This is why resume can be exact.

## 3. Atomic writes
A checkpoint is first written to a temporary file (`ckpt_00007.pt.tmp`), flushed to disk (**fsync**: force the operating system to really write it), and only then **renamed** to its final name (`ckpt_00007.pt`). A rename is atomic: at every moment the file either does not exist or is complete. A crash during writing leaves, at worst, an ignored `.tmp` file and the previous checkpoint intact. Only the newest `checkpoint.keep` (default 2) checkpoints are kept, and if the newest cannot be read, the loader falls back to the previous one.

## 4. Keeping the log consistent
`rounds.jsonl` is flushed and fsynced after every round, so it can contain rounds **newer than the last checkpoint** (for example, checkpoints every 2 rounds and a crash after round 3). The checkpoint stores the byte offset where the log ended at that moment. On resume, the log is **truncated** to that offset and the later rounds are recomputed. Because recomputation is deterministic, the recomputed rounds are the same ones, so no round appears twice. (The log is written in binary so that byte offsets mean the same on Windows and Linux.)

## 5. The configuration fingerprint
The checkpoint stores a **hash** (a short fixed-length fingerprint, here 16 hex characters of SHA-256) of the configuration. Resuming with a different configuration (another learning rate, seed, skew...) is **refused**, because it would silently mix two experiments. Two settings are deliberately left out of the fingerprint: `experiment.rounds` (so a finished run can be **extended** with more rounds) and the `checkpoint` settings (so you can change the cadence when resuming).

## 6. What each command does
| Existing run folder is... | no flag | `--resume` | `--overwrite` |
|---|---|---|---|
| absent | starts | starts | starts |
| completed | starts again (replaces it) | does nothing, returns the stored summary (or **extends** it if the config asks for more rounds) | starts again |
| unfinished (crashed, failed, killed) | **refuses**, tells you why | continues from the last checkpoint (or from scratch if none exists) | discards it and starts again |

It is always safe to add `--resume`, which is the habit to build on Colab. An unfinished run is never overwritten by accident.

## 7. Configuration keys
`checkpoint.every`: save after every N rounds (default 1; 0 = never, which also disables resume); the final round is always saved. `checkpoint.keep`: how many to keep (default 2). A checkpoint of round 0 (the untrained model) is written at the start.

## 8. How to run
**Laptop, interrupt a run on purpose:**
```
python -m pytest -q
python -m fedtune run configs/step5_tiny.yaml --out runs          # press Ctrl+C after 3 or 4 rounds
python -m fedtune run configs/step5_tiny.yaml --out runs          # refuses: unfinished run
python -m fedtune run configs/step5_tiny.yaml --out runs --resume # continues
```
Check the log has every round exactly once (replace the folder name with yours from `runs/`):
```
python -c "import json,glob; f=glob.glob('runs/*/rounds.jsonl')[0]; print([json.loads(l)['round'] for l in open(f)])"
```
Prove it equals an uninterrupted run: run once more with `--out runs_clean` without interrupting and compare the final `summary.json` metrics of both folders (they should agree within about 1e-4 on a CPU).

**Colab (GPU), the real use:** keep results on Google Drive, so they survive the session:
```
from google.colab import drive; drive.mount('/content/drive')
%cd fedtune
!python -m fedtune run configs/step4_real.yaml --out /content/drive/MyDrive/fedtune_runs --resume
```
If the session dies, reconnect, mount Drive again, run the **same cell**, and it continues.

## 9. What you should observe
- After Ctrl+C: the traceback ends with `KeyboardInterrupt`, and `summary.json` in the run folder says `"status": "failed"`.
- The second command prints the refusal; the third prints `resuming after round N (checkpoint)` and continues numbering from N+1.
- The round list has no gaps or repeats, and the final metrics match the uninterrupted run.

## 10. Things to understand for your viva
- Why does the checkpoint need no random-number state?
- Why write to a temporary file and rename?
- Why store the log offset, and what would go wrong without it?
- Why is resuming with a different configuration refused, and why is `rounds` excluded from the fingerprint?
- What is the difference between a skipped round and an interrupted run?

## 11. Limitations
Checkpoints are per round, so an interruption loses at most the round in progress. Resume is bit-exact on a CPU and equal within a small tolerance on a GPU (some GPU kernels are not deterministic). Checkpoints are meant to be loaded by the same code version. If Drive disconnects while saving, the run stops with an error and is resumable from the previous checkpoint.

## Next: Step 6
Client summaries and diversity-aware aggregation: the first of the proposed novelty strategies.
