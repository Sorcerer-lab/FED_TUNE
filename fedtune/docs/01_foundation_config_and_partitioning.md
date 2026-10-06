# Step 1: Foundation: configuration, seeding and data partitioning

**SRS coverage:** UC1, UC2, FR-1, FR-2, FR-3, FR-5, FR-6, FR-7.
**Goal of this step:** a validated configuration system and a reproducible way to split a dataset into skewed client shards, with a report that proves how skewed the split is. No model is trained yet.

## 1. Configuration (UC1, FR-1, FR-2)
**What:** all experiment settings live in one YAML file. Any value can be overridden on the command line, for example `--set dropout.rate=0.2`. The order is *defaults, then YAML file, then command-line overrides*.

**Why validate:** a typo like `dropout.rate: 1.4` would otherwise silently run a meaningless experiment for hours on Colab. Every key therefore has a type and an allowed range, and unknown keys are rejected (this catches typos such as `dropout.rat`). Errors name the key and the allowed range, as the SRS requires.

**Where:** `fedtune/config.py`. The `SCHEMA` table is the single place that defines what is allowed. Cross-checks live in `validate()` (for example, `min_participants` cannot exceed `num_clients`).

## 2. Named random streams (FR-3, FR-17)
**What:** one experiment seed feeds several independent random generators: `partition`, `dropout`, `model_init`, `training`, `summary`.

**Why this matters for the research:** to compare FedAvg, diversity-aware and clustered fairly, they must see the *same* data split and the *same* clients dropping in the *same* rounds. If everything shared one generator, switching strategy would change how many random numbers are consumed and silently change the split or the dropouts. Separate streams make that impossible. Technically: `np.random.SeedSequence([seed, stream_id])` produces statistically independent generators from one seed.

**Where:** `fedtune/seeding.py`. Stream ids are fixed forever, because renumbering them would invalidate old results.

## 3. Partitioning (UC2, FR-5)
The partitioner works only on **indices** (labels and group tags), never on text, so reports cannot leak content.

| Mode | Plain-words idea | Control knob |
|---|---|---|
| `dirichlet` (label skew) | For each class, deal its samples to clients using random proportions from Dirichlet(alpha). | `alpha`: 0.1 = extreme, 0.5 = strong, 5+ = nearly IID |
| `quantity` | Labels stay mixed evenly, but client sizes are very unequal. | `alpha` |
| `domain` / `language` | Each client has a "home" group. A fraction `heterogeneity` of each group's samples goes to its home clients; the rest is spread evenly. | `heterogeneity` in [0, 1]; 0 = IID, 1 = pure groups |

**Tiny example (dirichlet, 2 clients, one class of 100 samples):** with alpha=5 you might get a 55/45 split; with alpha=0.1 something like 96/4. Doing this for every class makes each client "specialise" in different classes.

**Safety net:** very small alpha can leave a client nearly empty. The code retries (up to 200 times) until every client has at least 10 samples, and otherwise stops with a message telling you to raise alpha.

## 4. Test and validation sets (FR-7)
1. A **global test set** is held out first, stratified (every class appears in it), and never used for training. It measures the merged model.
2. The remaining data is split across clients.
3. Each client then keeps a **local validation set** (default 10% of its shard) to measure per-client performance later.

Every sample ends up in exactly one place (test, one client's train, or one client's validation). A test checks this.

## 5. Heterogeneity measure and partition report (FR-6)
**Measure:** for each client we compare its label distribution to the global one using **JS divergence** (0 to 1), then take a size-weighted average. Result: 0 means IID, larger means more skewed. In domain/language mode we also report the same score over the group attribute.

**Why it matters:** it is the x-axis of our final heatmap. We can state "alpha=0.1 gave JS=0.4" rather than just "alpha=0.1".

**Report** (`partition_report.json`): per-client sizes, label counts, group counts, and the scores above.

## 6. How to run
```
pip install -r requirements.txt
python -m pytest -q                                   # 18 tests, about 1 second
python -m fedtune validate configs/smoke.yaml
python -m fedtune validate configs/smoke.yaml --set dropout.rate=1.4   # watch it get rejected
python -m fedtune demo-partition configs/smoke.yaml --set data.partition.alpha=5.0
python -m fedtune demo-partition configs/smoke.yaml --set data.partition.mode=domain --set data.partition.heterogeneity=0.9
```
`demo-partition` uses fixed **synthetic** labels (8000 samples, 4 classes); real datasets arrive in Step 2.

## 7. What you should observe (self-check)
- Same command twice gives an identical report; `--set experiment.seed=43` changes it.
- alpha 0.3 gives label JS of about 0.2; alpha 5.0 gives about 0.02.
- Domain mode with heterogeneity 0.9 gives group JS of about 0.38, with label JS near 0.
- `dropout.rate=1.4` exits with an error naming `dropout.rate`.

## 8. Things to understand for your viva
- Why use separate random streams? (fair comparison across strategies)
- Why Dirichlet and not "give each client 2 classes"? (it gives a continuous dial, and it is the standard in the literature, e.g. Hsu et al. 2019, already in your SRS references)
- What does the JS number mean, and why is it size-weighted?
- Why does the report contain counts but never text? (NFR-6 privacy boundary)

## 9. Known limitations
Synthetic partitions imitate real non-IID data and need not match real institutions (SRS 4.9). The language mode reuses the domain mechanism, so it needs a dataset that actually carries a language tag (Step 2 will pick one).

## Next: Step 2
Dataset loading (AG News plus a domain/language-tagged dataset), tokenisation, the LoRA client trainer and the evaluator. Its reference doc will come first.
