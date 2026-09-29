import os
import statistics

import torch

from Checkpointing import checkpoint_path
from GNNNet import GNNNet
from Paths import paths_for

DATASET = "davis"
FOLDS = [0, 1, 2, 3, 4]


def main() -> None:
    paths = paths_for(DATASET)
    print(f"Checkpoints: {paths.models}")
    print(f"{'fold':>4}  {'last_epoch':>10}  {'best_epoch':>10}  {'best_mse':>10}")

    best_epochs = []
    for fold in FOLDS:
        path = checkpoint_path(paths.models, GNNNet.__name__, DATASET, fold)
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
