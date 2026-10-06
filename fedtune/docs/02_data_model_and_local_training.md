# Step 2: Real data, the base model with LoRA, local training and evaluation

**SRS coverage:** FR-4, FR-9, FR-10, FR-24 (accuracy and macro-F1), NFR-2 (one shared base model), NFR-3 (upload size).
**Goal:** one simulated client can load its data shard, fine-tune a LoRA adapter on it, and be evaluated. **There is no merging yet** (that is Step 3).

> **Testing note:** I could run the data, tokenizer, metrics and config parts here (all pass). I could NOT run the PyTorch/PEFT parts (the model, the client trainer, the evaluator), because my sandbox cannot install torch or download models. Those have a test file, `tests/test_step2_torch.py`, which you run yourself. If anything fails, paste the error and I'll fix it.

## 1. Datasets (FR-4)
| Name | Task | Group used for partitioning | Why it is here |
|---|---|---|---|
| `ag_news` | news topic classification, 4 classes | none (label skew only) | simple, fast, standard benchmark |
| `multi_nli` | sentence-pair inference (does sentence B follow from A?), 3 classes | **genre** (fiction, government, slate, telephone, travel) | real *domain* shift with the same task |
| `xnli` | the same inference task in several languages (en, fr, de, es, hi) | **language** | real *language* shift; needs a multilingual model |
| `synthetic` | made-up words with class and group signal | 4 groups | offline tests, no downloads |

`data.max_samples` caps how many samples are used (so runs fit on a free GPU). The subset is chosen with its own random stream (`data`), so it is reproducible and never disturbs the partition.

## 2. Tokenisation (turning text into numbers)
A **tokenizer** splits text into **tokens** (often word pieces) and maps each to an integer id, because models only read numbers. Steps:
1. **Truncation:** cut every sample to `data.max_length` tokens (default 128) to bound memory and time.
2. **Dynamic padding:** samples in a batch have different lengths, so we pad the shorter ones to the longest *in that batch* with a special pad id.
3. **Attention mask:** 1 for real tokens, 0 for padding, so the model ignores the padding.
4. **Pairs:** for sentence-pair tasks both sentences go in one sequence separated by a special separator token.

We tokenise **once** for the whole dataset (`data/encoding.py`) and keep the ids; each client later just picks its rows by index.

## 3. The base model and LoRA
- **Base model:** a pretrained *encoder* (default `distilbert-base-uncased`, 66M parameters) with a **classification head** on top (a small layer that turns the model's summary of the text into class scores). The base weights are **frozen**.
- **LoRA (Low-Rank Adaptation):** for chosen weight matrices `W`, the layer computes `W x + (alpha / r) * B A x`. `A` and `B` are thin matrices of **rank** `r` (default 8). Only `A` and `B` are trained. `alpha / r` is the **scaling** that controls how strongly the correction acts.
  - `B` starts at zero, so at the beginning the adapted model behaves **exactly** like the base model. A test checks this.
  - **Target modules:** which matrices get a LoRA correction. By default the attention *query* and *value* projections (for DistilBERT they are called `q_lin` and `v_lin`; the code auto-detects names for BERT, RoBERTa, XLM-R, DeBERTa-v2 and others, or you set `model.lora.target_modules`).
- **What is "the update"?** The LoRA matrices **plus the classification head**. The head must be shared too: otherwise each client would learn its own private head and the merged adapters would not fit any head. We call this the **trainable state**. It is the only thing a client uploads and the server sends back.
- **Initial global state:** one seed (`model_init` stream) fixes the starting head and `A` matrices, so every client begins from the identical global model.

`python -m fedtune local-check ...` prints the exact trainable-parameter count and the upload size as a percentage of the full model (SRS NFR-3 target: at most 5%). For DistilBERT it should be far below that, but read the printed number instead of trusting me.

## 4. One shared base model for all clients (NFR-2)
We hold the big base model **once in memory**. To "become" client k, we copy the global trainable state into the model, train, and read the trainable state back out (`model.py: set_trainable_state / get_trainable_state`). Memory therefore does not grow with the number of clients.

## 5. Local training (`client.py`, FR-9)
For each client: load global state, then for `train.local_epochs` passes over its shard, in batches of `train.batch_size`:
1. forward pass gives a **loss** (how wrong the predictions are; **cross-entropy**),
2. **backpropagation** computes gradients for the trainable parameters only,
3. **gradient clipping** (max norm 1.0) stops rare huge gradients from wrecking training,
4. the **AdamW optimiser** updates the parameters with learning rate `train.lr`.

A **fresh optimiser is created every round**, which is standard in federated learning (optimiser memory is not shared). Randomness (batch order, LoRA dropout) is seeded from `(seed, round, client)`, so the same call gives the same result. A test checks this on CPU.

## 6. Evaluation (`evaluate.py`, `metrics.py`)
- **Accuracy:** share of correct predictions.
- **Macro-F1:** per-class F1, averaged, so rare classes count equally. We use it because under label skew accuracy can look fine while a class is ignored.
- **Loss** is also reported. The **global test set** judges the shared model; each client's **local validation set** judges its personal fit.

## 7. How to run
**A) Offline pipeline check on your laptop (no downloads, a few seconds):**
```
python -m pip install -r requirements.txt
python -m pytest -q
python -m fedtune local-check configs/tiny.yaml
```
Expected: the torch tests pass; the table shows loss falling and accuracy above 0.25 (chance with 4 classes). The model here is a tiny random one, so it only proves the plumbing works.

**B) Real check on Colab (GPU runtime):**
```
!git clone <your-repo-url> && cd fedtune     # or upload the zip and unzip it
%cd fedtune
!pip -q install peft datasets
!python -m pytest -q
!python -m fedtune local-check configs/step2_real.yaml --max-clients 3
```
Expected: the first run downloads DistilBERT and AG News; "own-val acc" should clearly beat 0.25 chance; "global-test acc" is usually **lower** than "own-val acc" under strong skew.

## 8. What you should observe, and why it matters
- Each client's accuracy on its **own** validation set is higher than on the **global** test set. That gap is **client drift**: a client trained on mostly one class forgets the others. It is the core problem that diversity-aware and clustered aggregation try to fix, and it is what you will explain in the viva.
- Smaller `data.partition.alpha` widens the gap.
- Upload size is a tiny fraction of the full model.

## 9. Things to understand for your viva
- Why freeze the base model and train only LoRA? (cost and communication)
- Why is `B` initialised to zero?
- Why must the classification head be part of the shared state?
- What is the difference between a client's validation set and the global test set?
- Why create a new optimiser each round?

## 10. Known limitations
- Full 32-bit precision for simplicity (mixed precision could speed up Colab later).
- `token_type_ids` (segment markers) are not passed; fine for DistilBERT and RoBERTa-style models.
- Dataset hub names change over time; the loader tries both the new and the legacy ids, and prints every error if all fail.
- XNLI downloads five languages, so the first load takes a while.

## Next: Step 3
The round loop: send the global state to clients, collect their updates, merge with FedAvg (weighted by data size), evaluate, log every round as JSONL, and account for bytes sent. Its reference doc comes first.
