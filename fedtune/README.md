# FedTune
Federated LoRA fine-tuning simulator. Read `docs/00_roadmap_and_glossary.md` first, then the numbered step docs.
Status: Steps 1-6 (config, partitioner, data, LoRA model, FedAvg round loop, dropout, network model with deadline, checkpoint and resume, client summaries, diversity-aware aggregation).
Run from this folder: `python -m pytest -q`, `python -m fedtune diversity-demo configs/smoke.yaml`, `python -m fedtune run configs/step6_tiny.yaml`
