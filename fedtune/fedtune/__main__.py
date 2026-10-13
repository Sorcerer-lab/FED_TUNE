"""Command line: python -m fedtune {validate,demo-partition,local-check,network-demo,diversity-demo,run} ..."""
import argparse
import json
import sys

import numpy as np

from .config import ConfigError, load_config
from .data.partition import PartitionError, build_partition, write_report


def _common(p):
    p.add_argument("config", nargs="?", default=None, help="path to a YAML config (optional)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override a key, e.g. --set dropout.rate=0.2")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fedtune")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _common(sub.add_parser("validate", help="check a configuration and print it"))
    d = sub.add_parser("demo-partition", help="partition a synthetic dataset and print the report")
    _common(d)
    d.add_argument("--out", default="partition_report.json")
    lc = sub.add_parser("local-check", help="Step 2: train each client once and evaluate (needs torch)")
    _common(lc)
    lc.add_argument("--max-clients", type=int, default=None, help="only train the first N clients")
    nd = sub.add_parser("network-demo", help="Step 4: show participation, dropout and network times (no training, no torch)")
    _common(nd)
    nd.add_argument("--payload-mb", type=float, default=2.964, help="size of one model update in MB (default: DistilBERT LoRA+head)")
    dd = sub.add_parser("diversity-demo", help="Step 6: show coverage gains and diversity weights on synthetic embeddings (no torch)")
    _common(dd)
    rn = sub.add_parser("run", help="Step 3: federated training with every strategy in the config (needs torch)")
    _common(rn)
    rn.add_argument("--out", default="runs", help="folder for run logs (default: runs)")
    rn.add_argument("--resume", action="store_true", help="continue an unfinished run from its last checkpoint (safe to always add)")
    rn.add_argument("--overwrite", action="store_true", help="start an unfinished run over, discarding its logs and checkpoints")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config, args.overrides)
        if args.cmd == "validate":
            print("config OK")
            print(json.dumps(cfg, indent=2))
            return 0
        if args.cmd == "network-demo":
            from .network_demo import network_demo
            network_demo(cfg, args.payload_mb)
            return 0
        if args.cmd == "diversity-demo":
            from .diversity_demo import diversity_demo
            diversity_demo(cfg)
            return 0
        if args.cmd == "run":
            from .runner import run_all
            run_all(cfg, args.out, resume=args.resume, overwrite=args.overwrite)
            return 0
        if args.cmd == "local-check":
            from .pipeline import run_local_check
            run_local_check(cfg, args.max_clients)
            return 0
        rng = np.random.default_rng(0)  # fixed synthetic data, independent of experiment seed
        labels = rng.integers(0, 4, size=8000)
        groups = rng.integers(0, 4, size=8000)
        part = build_partition(labels, cfg, groups=groups)
        write_report(part.report, args.out)
        r = part.report
        print(f"mode={r['mode']} clients={r['n_clients']} test={r['n_test']} "
              f"size range=[{r['size_min']}, {r['size_max']}] "
              f"label JS={r['label_heterogeneity_js']:.3f} group JS={r['group_heterogeneity_js']:.3f}")
        print(f"report written to {args.out}")
        return 0
    except (ConfigError, PartitionError, RuntimeError) as e:  # PipelineError, DataError are RuntimeErrors
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
