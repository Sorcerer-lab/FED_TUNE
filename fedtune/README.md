# FedTune
Federated LoRA fine-tuning simulator. Read `docs/00_roadmap_and_glossary.md` first, then the numbered step docs.
Status: Steps 1-3 (config, partitioner, data, LoRA model, local training, round loop with FedAvg, logging).
Run from this folder: `python -m pytest -q`, `python -m fedtune run configs/tiny.yaml`
