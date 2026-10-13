# Step 6: Client summaries and diversity-aware aggregation (first novelty strategy)

**Design coverage:** bounded client summaries (privacy boundary), diversity-aware weights (Algorithm 4 of the design document), the "coverage compensation" variant for dropout.
**Goal:** replace "weight = amount of data" (FedAvg) by "weight = amount of data × how much *new* coverage this client adds", without ever sending raw text or labels to the server.

> **Testing note:** the k-means, summaries, coverage gains and weights are tested here (87 tests pass, 30 for Steps 6). The embedding step and the full runs need PyTorch, which my sandbox cannot run, so `tests/test_step6_torch.py` is for you to run. Paste any error and I'll fix it.

## 1. The idea in plain words
FedAvg trusts volume: the client with most data counts most. But ten clients holding near-identical data add little over one of them, while a small client with rare content adds something nobody else has. Diversity-aware aggregation tries to give weight to **coverage** of the data space instead of only to size. The server must judge coverage without seeing data, so each client sends a small **summary**.

## 2. The client summary
1. **Embedding.** An **embedding** is a vector of numbers that represents the meaning of a text; similar texts have nearby vectors. Each client embeds its own training texts with the **frozen pretrained encoder** (mean of the last layer's hidden states). At setup every LoRA matrix B is zero, so the adapted model equals the pretrained one: the embeddings depend only on the pretrained encoder and never change during training. That is why the summary is computed once.
2. **Clustering.** The client groups its embeddings into K = 8 groups with **k-means** (repeat: assign each point to its nearest centre, move each centre to the mean of its points).
3. **What is sent:** the K centres (unit-length vectors) and how many samples fell in each. Size = K × d × 4 + K × 4 bytes = **24,608 bytes** for DistilBERT (d = 768), about 0.8% of a single model upload, sent once.
4. **What it reveals (stated, not hidden):** approximate cluster centres of the client's data in embedding space and the size of each cluster. **No text, no token ids, no per-example labels.** The exact sizes and this statement are written to `summary_log.json` in every run folder, and the bytes are counted in the cumulative upload.

## 3. Coverage and the gain of one client
Let U be the pool of all centres (each with its count m), and S the participating clients' centres. **Cosine similarity** measures how alike two directions are (1 = same, 0 = unrelated). The **coverage** of S is
```
f(S) = sum over centres u in U of  m_u * max over centres s in S of cos(u, s)
```
("each piece of data, weighted by size, is credited with how well its closest participating centre matches it".) The **leave-one-out gain** of client k is `f(S) − f(S without k)`: how much coverage is lost if k is removed.

**Toy example (2-D):** clients A and B both have data in direction (1, 0); client C has data in direction (0, 1). Removing A loses nothing, because B still covers it, so gain(A) = 0; same for B; removing C loses all of C's data, so gain(C) is large. C gets extra weight, A and B do not.

## 4. From gain to weight
```
c_k = (gain_k / n_k), then scaled so that the mean over participants is 1
w_k proportional to  n_k^beta * (c_k + epsilon)^gamma     (then normalised; optional cap)
```
- **Why divide by n_k ("per_sample"):** my first version did not, and the demo showed the flaw. The gain grows with the number of samples a client holds, so multiplying it by n_k counted volume twice (about n²). A small client holding one rare class (298 samples) got weight **0.003**, *below* its FedAvg weight of 0.021. Dividing by n_k measures *uniqueness per sample* instead. `diversity.gain: total` keeps the old behaviour as an ablation.
- **Demo, strong skew (alpha 0.1, 6 clients):**

| Client | Samples | Data | FedAvg weight | diversity (per_sample) | diversity (total) |
|---|---|---|---|---|---|
| 0 | 298 | 100% class 0 | 0.021 | **0.040** | 0.003 |
| 1 | 160 | mostly class 1 | 0.011 | **0.019** | 0.001 |
| 3 | 2,623 | 95% class 2 | 0.182 | 0.243 | 0.138 |
| 5 | 7,475 | mixed | 0.519 | **0.427** | 0.692 |

  Small, unusual clients gain weight; the large mixed client loses some. Run it yourself (section 8).
- **Special cases:** `gamma = 0` is exactly FedAvg (the built-in ablation). `beta = 0` ignores volume (with `gain: total`). `epsilon` keeps every weight positive. `max_weight` (for example 0.5) caps any single client; if a cap is impossible it is raised to 1/m.

## 5. Gains are recomputed for whoever participates
Each round the gains are computed over the **participants only**, so dropout and deadlines change the weights automatically.
`diversity.universe: all` is the **coverage compensation** variant: the pool U includes the centres of clients that are *absent*, so a missing client's data counts as uncovered and the participants most similar to it gain weight. It is one config switch, so it is already available for the dropout experiments.

## 6. Honest limitations
- **Duplicates:** if two clients hold the same kind of data, each looks redundant given the other (both gain 0), although one of them is needed. `epsilon` and the volume factor keep both in play, but this is a known weakness of leave-one-out.
- Embeddings from a general encoder may separate *style* more than *label*, so "coverage" is only a proxy; K = 8 is coarse.
- Nothing guarantees an improvement. Hypothesis H2 is to be tested, and a result where FedAvg wins is reported as found.
- The summary leaks more than nothing (section 2.4); no formal privacy is claimed.

## 7. What is logged
`client_summaries.npz` and `summary_log.json` per run; round 0 counts the summary bytes as upload (`up_summary`); every round has `aggregator_info` with `coverage_gain` (scaled), `gain_raw`, `universe`, `gain`; weights are in `aggregation_weights`. A resumed run loads the saved summaries instead of recomputing them.

## 8. How to run
**Laptop, no torch, seconds** (see the weights and how they change):
```
python -m fedtune diversity-demo configs/smoke.yaml --set data.partition.alpha=0.1 --set experiment.num_clients=6
python -m fedtune diversity-demo configs/smoke.yaml --set data.partition.alpha=0.1 --set experiment.num_clients=6 --set diversity.gain=total
python -m fedtune diversity-demo configs/smoke.yaml --set data.partition.alpha=0.1 --set experiment.num_clients=6 --set diversity.max_weight=0.4
```
**Laptop, full pipeline on the tiny model:**
```
python -m pytest -q
python -m fedtune run configs/step6_tiny.yaml --out runs
```
It runs FedAvg and diversity-aware one after the other on the same partition and participation, and prints one comparison table. The tiny random model learns little, so judge only that it runs and the files appear.

**Colab (GPU):**
```
%cd fedtune
!pip -q install peft datasets
!pip uninstall -y torchao
!python -m pytest -q
!python -m fedtune run configs/step6_real.yaml --out /content/drive/MyDrive/fedtune_runs --resume
```
(mount Drive first, as in the Step 5 doc). Expected: a first line `computing client summaries ...` (about a minute), then 10 rounds per strategy and a table with both strategies.

## 9. What to look for, and experiments worth running
- The comparison table: final accuracy, macro-F1 and **worst-client accuracy** for FedAvg versus diversity-aware. Differences may be small or negative; one seed is not evidence, so the grid (Step 10) will use three seeds.
- In `rounds.jsonl`, compare `aggregation_weights` of the two runs: which clients does diversity-aware up-weight?
- Variations to try later through `--set`: `diversity.gamma` 0 / 0.5 / 1 / 2, `diversity.gain` per_sample versus total, `diversity.universe` participants versus all together with `dropout.rate=0.3`, `diversity.max_weight=0.5`.

## 10. Things to understand for your viva
- What is an embedding, and why are embeddings from the frozen encoder fixed during training?
- What exactly does the server learn from a summary, and what does it not learn?
- What is the leave-one-out gain, and what happens to two identical clients?
- Why divide the gain by the client's size? What went wrong without it?
- Which settings make the strategy collapse to FedAvg, and why is that useful?

## Next: Step 7
Update-similarity **clustered** aggregation, and the **hybrid** that clusters clients by their LoRA updates and then weights the clusters by diversity.
