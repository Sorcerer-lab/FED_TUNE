# FedTune
Federated LoRA fine-tuning simulator. Read `docs/00_roadmap_and_glossary.md` first, then the numbered step docs.
Status: Steps 1-4 (config, partitioner, data, LoRA model, local training, round loop with FedAvg, dropout, network model with deadline, failure handling).
Run from this folder: `python -m pytest -q`, `python -m fedtune network-demo configs/step4_real.yaml`, `python -m fedtune run configs/step4_tiny.yaml`
