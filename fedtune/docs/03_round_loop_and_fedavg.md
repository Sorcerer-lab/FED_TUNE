# Step 3: The federated round loop, FedAvg, logging and communication accounting

**SRS coverage:** UC3, UC5 (same partition and seed for every strategy), FR-13, FR-17, FR-24, FR-25 (communication), NFR-3.
**Goal:** run real federated training: clients train, the server merges with FedAvg, the global model is evaluated every round, and everything is logged. Dropout is **not** simulated yet (Step 4), so all clients take part in every round.

> **Testing note:** the aggregation maths, logging and summary are tested here (40 tests pass). The full round loop needs PyTorch, which my sandbox cannot run, so `tests/test_step3_torch.py` is for you to run. Paste any error and I'll fix it.

## 0. What your Step 2 Colab output told us
- **The real-model path works** (model load, LoRA targets, dataset download, GPU). The "MISSING / UNEXPECTED" report is normal: the classification head is new, and DistilBERT's language-model head is unused.
- **The upload is 741,124 numbers. Only 147,456 are LoRA; about 80% is the classification head.** (LoRA: 6 layers × 2 matrices × 12,288 = 147,456. Head: `pre_classifier` 590,592 + `classifier` 3,076.) So the communication saving mostly comes from freezing the encoder, not from LoRA itself. From now on every round logs LoRA bytes and head bytes separately (`up_lora`, `up_head`), so the report can state both honestly. Whether to freeze `pre_classifier` to shrink the head is an open design choice; I recommend keeping the current setup and deciding when we see results.
- **Client drift is visible:** client 2 scored 0.981 on its own data but 0.251 on the global test set. This is exactly the failure that better aggregation tries to fix.
- **Client sizes vary a lot** (50 to 1,438 samples). A 50-sample client does only 4 training steps, and under FedAvg it gets a tiny weight (about 3%).

## 1. One round, step by step (`runner.py`)
1. **Select participants** (`participation.py`; everyone for now).
2. **Broadcast:** every participant starts from the current global trainable state (download).
3. **Local training:** each runs `train_client` (Step 2) on its own shard.
4. **Upload:** each returns its updated trainable state.
5. **Aggregate:** the server merges the updates into a new global state.
6. **Evaluate:** the new global model is scored on the global test set and on every client's validation set.
7. **Log** one JSON line.

Round **0** is logged before any training. It records the untrained model, so curves start from the true baseline.

## 2. FedAvg (`aggregation/fedavg.py`, FR-13)
For participating clients k with `n_k` training samples:
```
w_k = n_k / (sum of all n)          new_parameter = sum over k of  w_k * parameter_k
```
It is a weighted average **of every trainable tensor separately**: the LoRA matrices `A`, the LoRA matrices `B`, and the head. Clients with more data count more. The weights always sum to 1, so a subset of clients is renormalised automatically (FR-20).

**Important caveat (SRS 4.9):** the update that LoRA actually applies is `B x A`. Averaging `A` and `B` separately gives `avg(B) x avg(A)`, which is **not** the average of the products `avg(B x A)`, because cross terms appear. This is a known LoRA aggregation problem. Step 8 adds methods that handle it (stacking, product averaging with re-factorisation), and we will state which one every result uses (FR-16).

## 3. Strategy interface (`aggregation/base.py`)
A strategy only has to say **how much each client counts** (`weights`). Strategies that need more, like clustering, override `aggregate`. The registry (`aggregation/__init__.py`) already knows the later strategies; asking for one before its step gives a clear message ("planned for Step 6").

## 4. Communication accounting (FR-24, FR-25, NFR-3)
- **up** = total bytes of the trainable tensors uploaded by participants that round. **down** = global-state bytes × number of participants.
- Counted at the tensors' stored precision (32-bit), without network protocol overhead. Fine for comparing strategies and LoRA settings.
- `up_lora` and `up_head` split `up` (see section 0). `comm_cum_bytes` is the running total.

## 5. What is logged (`runlog.py`)
Folder `runs/<run_id>/`, where the run id looks like `ag_news-dirichlet0p3-drop0p0-fedavg-s42`:
- `rounds.jsonl`: one JSON object per round, flushed immediately, so a crash never loses finished rounds (NFR-7). Fields: participating and dropped clients, aggregation weights, global metrics, per-client metrics, per-client training loss, bytes, round time.
- `summary.json`: rounds completed and skipped, final and initial metrics, per-client accuracy mean, std, worst and best (`min` is the "worst client"), cumulative bytes, total time, peak GPU memory.
- `config.json`: the exact configuration, enough to regenerate the partition. `meta.json` and `partition_report.json` complete the record.
If `experiment.strategy` lists several strategies, they run **one after another on the same partition, seed and starting model**, which makes the comparison fair (UC5).

## 6. How to run
**Laptop, offline plumbing check (about a minute):**
```
python -m pytest -q
python -m fedtune run configs/tiny.yaml --out runs
```
Expected: 3 round lines plus a summary table, and about 42 passing tests. The tiny model will not learn much, so judge only that it runs and the files appear.

**Colab (GPU):**
```
%cd fedtune        # after unzipping the new zip
!pip -q install peft datasets
!pip uninstall -y torchao
!python -m pytest -q
!python -m fedtune run configs/step3_real.yaml --out runs
```
Expected: 8 rounds, with global accuracy rising from about 0.25 and per-round time of roughly 1 to 2 minutes. Look at `runs/*/rounds.jsonl` and `summary.json`.

## 7. What you should observe
- Global accuracy and macro-F1 climb over rounds, but noisily, because `alpha=0.3` is strongly skewed.
- Re-run with `--set data.partition.alpha=5.0`: it should converge faster and higher. This gap is the problem your novelty targets.
- `up_lora` is about 0.59 MB per client per round, and `up_head` about 2.4 MB.

## 8. Things to understand for your viva
- Why weight by data size, and when is that unfair? (a client with lots of redundant data dominates)
- Why does averaging `A` and `B` separately not equal averaging the update?
- Why log round 0 and flush every line?
- What do "up" and "down" count, and what do they leave out?

## 9. Known limitations
No dropout yet. No checkpointing yet (Step 5), so a Colab disconnect loses the run. Peak memory is only measured on GPU.

## Next: Step 4
The dropout simulator (independent per-round dropout with probability p), the minimum-participants rule and skipped rounds, participation logging, non-finite-update handling, and (if you want it) the network model with bandwidth, latency and a round deadline.
