# FedTune: Roadmap and Glossary

Read this once. Every later step has its own reference doc, which you read **before** its code.

## The project in plain words
Imagine 10 hospitals that each want a better text classifier but cannot share their documents. Each hospital trains a small add-on to a shared language model on its own data and sends **only that add-on** to a central server. The server merges the add-ons into one improved add-on and sends it back. This repeats for several rounds.

Two things make this hard, and they are what FedTune studies:
1. Hospitals have **different kinds of data** (non-IID), so a naive merge gets pulled in conflicting directions.
2. Some hospitals **drop out** of some rounds, so the merge must cope with missing contributors.

Our novelty is a smarter merge: weight each contributor by how much *new coverage* its data adds (**diversity-aware**), group contributors whose updates point in similar directions (**clustered**), and combine both (**hybrid**). We test them against plain averaging and against an existing method (FedSA-LoRA), over a grid of skew levels and dropout rates.

## Roadmap (nothing is cut)
| Step | What we build | SRS items | Doc |
|---|---|---|---|
| 1 | Project skeleton, config + validation, seeding, partitioner, partition report | UC1, UC2, FR 1-3, 5-7 | `01_...` (ready) |
| 2 | Dataset loading, tokenisation, LoRA client trainer, evaluator | FR 4, 9, 10, 24 | `02_...` (ready) |
| 3 | Round loop with FedAvg, JSONL logging, communication accounting | UC3, FR 13, 24, 25 | `03_...` (ready) |
| 4 | Dropout simulator, minimum participants, skipped rounds, broken-update handling, network model with deadline | UC4, UC8, UC12 | `04_...` (ready) |
| 5 | Checkpoint and resume | UC9, FR 27, 34 | next |
| 6 | Client summaries and diversity-aware aggregation | FR 11-14 | |
| 7 | Clustered aggregation and the hybrid | FR 15, 15a | |
| 8 | LoRA merge methods (averaging, stacking, product + re-factorisation) and FedSA-LoRA baseline | FR 16 | |
| 9 | Plug-in strategy interface, stale-update reuse, dropout-compensation extension | UC11, FR 18, 22 | |
| 10 | Grid runner, seeds, fair tuning protocol | UC5, UC6, FR 26 | |
| 11 | Dashboard | UC7, FR 29-32 | |
| 12 | Export, unit tests, integration test, results writeup | UC10, FR 28, NFR 10 | |

**Honest limit:** the SRS capacity targets (50 clients, 100 rounds, about 1B-parameter models) depend on GPU time. We build the code to support them, but we will only run them if Colab/Kaggle hours allow.

## Glossary
- **Federated learning (FL):** training one shared model where data stays on each client and only model updates travel.
- **Client / coordinator (server):** a data holder that trains locally / the central party that merges updates.
- **Round:** one cycle of send model, train locally, send updates, merge.
- **Non-IID:** data that is **not** independent and identically distributed, meaning clients see different slices of the world (different classes, topics, languages, or amounts).
- **Label skew:** clients have different mixes of classes. **Quantity skew:** clients have very different amounts of data. **Domain / language skew:** clients come from different topics or languages.
- **Dirichlet distribution, parameter alpha:** a recipe for random proportions that sum to 1. Small alpha (0.1) gives lopsided proportions (a client holds mostly one class); large alpha (5+) gives near-equal proportions.
- **LoRA (Low-Rank Adaptation):** instead of changing all weights of a big model, learn a small correction `delta W = B x A`, where A and B are thin matrices. The big model stays frozen. **Rank** `r` is how thin they are (smaller = fewer parameters). The learned A and B together are the **adapter**.
- **FedAvg:** the baseline merge, a weighted average of client updates, weighted by how much data each client has.
- **Aggregation:** the merge step on the server.
- **Dropout / partial participation:** a client misses a round. We model it as an independent coin flip with probability p.
- **Seed / reproducibility:** a number that fixes all randomness so a run can be repeated exactly.
- **Jensen-Shannon (JS) divergence:** a number from 0 to 1 measuring how different two distributions are (0 = identical).
- **Macro-F1:** an accuracy-like score that averages per-class F1, so rare classes count as much as common ones.
- **Checkpoint:** a saved snapshot of a run so it can resume after an interruption.
- **Embedding:** a vector of numbers that represents the meaning of a text; similar texts get nearby vectors.
- **Clustering:** automatically grouping similar items (here, clients with similar update directions).
- **Tokenizer / token:** splits text into pieces (tokens) and maps each to an integer id that the model can read.
- **Attention mask:** marks which positions are real tokens (1) and which are padding (0).
- **Classification head:** a small final layer that turns the model's summary of a text into class scores.
- **Frozen:** parameters that are not updated during training.
- **Loss / cross-entropy:** a number measuring how wrong the predictions are; training lowers it.
- **Optimiser (AdamW), learning rate:** the rule and step size used to update parameters from gradients.
- **Gradient clipping:** capping the gradient size so one bad batch cannot wreck training.
- **Client drift:** a client's model moving toward its own data's quirks, away from what is good for everyone.
- **JSONL:** a text file with one JSON object per line; easy to append to and safe to read after a crash.
- **Upload / download bytes:** size of the trainable tensors sent from clients to the server / from the server to clients.
- **Bandwidth (Mbit/s):** how many millions of bits per second a link carries; 8 bits make one byte.
- **Latency:** the fixed delay of one transfer, independent of its size.
- **Straggler:** a client too slow to finish within the round deadline.
- **Deadline:** the longest the server waits in a round; later updates are ignored.
- **Simulated time:** time computed from formulas (not measured with a clock), so it is identical for every strategy.
- **Jitter:** small random variation of bandwidth from round to round.
- **Skipped round:** a round with too few valid participants; the global model does not change.
- **Non-finite update:** an update containing NaN (not a number) or Inf (infinity), usually from numeric overflow.
