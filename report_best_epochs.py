import argparse
import json
import os
import statistics

import torch

from Checkpointing import checkpoint_path
from Paths import paths_for
from RunTag import MODEL_CLASS_NAMES, add_run_args, check_run_args, run_tag

DATASET = "davis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the best epoch of each fold checkpoint."
    )
    add_run_args(parser)
    args = parser.parse_args()
    check_run_args(parser, args)
    return args


def main() -> None:
    args = parse_args()
    paths = paths_for(DATASET, args.pdb_set)
    model_name = MODEL_CLASS_NAMES[args.protein_repr]
    tag = run_tag(
        args.protein_repr, args.plm_model, args.seed, args.standardize, args.pdb_set
    )
    with open(paths.train_folds) as file:
        n_folds = len(json.load(file))
    print(f"Checkpoints: {paths.models}")
    print(f"{'fold':>4}  {'last_epoch':>10}  {'best_epoch':>10}  {'best_mse':>10}")

    best_epochs = []
    for fold in range(n_folds):
        path = checkpoint_path(paths.models, model_name, DATASET, fold, tag)
        if not os.path.exists(path):
            print(f"{fold:>4}  missing {path}")
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        best_epochs.append(checkpoint["best_epoch"])
        print(
            f"{fold:>4}  {checkpoint['epoch']:>10}  {checkpoint['best_epoch']:>10}  "
            f"{float(checkpoint['best_mse']):>10.4f}"
        )

    if not best_epochs:
        raise SystemExit("No fold checkpoints found.")
    print(f"median best_epoch: {statistics.median(best_epochs)}")


if __name__ == "__main__":
    main()
