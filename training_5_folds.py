import argparse
import gc
import json
import os
from collections.abc import Callable

import torch
from torch.utils.data import Dataset

from src.common.checkpointing import checkpoint_path, load_checkpoint, save_checkpoint
from src.common.conditions import Condition, build_condition
from src.common.creating_train_and_test_set import load_fold_indices, load_test_indices
from src.common.dta_dataset import DataLoader, predicting, train
from src.common.emetrics import get_mse
from src.common.paths import DatasetPaths, paths_for
from src.common.run_tag import add_run_args, check_run_args, run_tag, tag_suffix
from src.common.seeding import set_seed

dataset = "davis"

# Batch sizes, LR and epoch count as in DGraphDTA (Jiang et al., RSC Advances 2020) training_5folds.py.
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.001
NUM_EPOCHS = 2000
# Checkpoint cadence in epochs.
CHECKPOINT_EVERY_EPOCHS = 10
EARLY_STOP_PATIENCE = 100
# Checkpoint identity for the full-data model, outside the CV fold range 0-4.
FULL_FOLD = -1
FULL_TRAIN_ROWS = 25046


def make_train_loader(
    train_data: Dataset, seed: int, collate_fn: Callable
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        train_data,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )


def patience_exhausted(epoch: int, best_epoch: int) -> bool:
    return best_epoch > 0 and epoch - best_epoch >= EARLY_STOP_PATIENCE


def record_best_epoch(
    results_dir: str,
    fold: int,
    best_epoch: int,
    best_mse: float,
    last_epoch: int,
    tag: str,
) -> None:
    path = os.path.join(results_dir, f"best_epochs{tag_suffix(tag)}.json")
    records = {}
    if os.path.exists(path):
        with open(path) as file:
            records = json.load(file)
    records[str(fold)] = {
        "best_epoch": best_epoch,
        "best_val_mse": float(best_mse),
        "last_epoch": last_epoch,
        "stopped_early": last_epoch < NUM_EPOCHS,
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as file:
        json.dump(records, file, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def run_fold(
    fold: int,
    condition: Condition,
    tag: str,
    seed: int,
    paths: DatasetPaths,
    device: torch.device,
) -> None:
    set_seed(seed)
    model = condition.build_model()
    model.to(device)
    model_st = condition.model_name
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    checkpoint_file_name = checkpoint_path(paths.models, model_st, dataset, fold, tag)
    start_epoch, best_mse, best_epoch = load_checkpoint(
        checkpoint_file_name, fold, model, optimizer, device
    )
    if start_epoch >= NUM_EPOCHS or patience_exhausted(start_epoch, best_epoch):
        print(
            f"Fold {fold} already finished at epoch {start_epoch} "
            f"(best MSE {best_mse} at epoch {best_epoch}), skipping"
        )
        record_best_epoch(paths.results, fold, best_epoch, best_mse, start_epoch, tag)
        return
    if start_epoch > 0:
        print(
            f"Resuming fold {fold} at epoch {start_epoch + 1} (best MSE {best_mse} at epoch {best_epoch})"
        )

    print(f"Training for fold {fold}...")

    train_data, valid_data = condition.fold_datasets(fold)
    train_loader = make_train_loader(train_data, seed + fold, condition.collate)
    valid_loader = DataLoader(
        valid_data,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        collate_fn=condition.collate,
    )

    model_file_name = os.path.join(
        paths.models, f"model_{model_st}_{dataset}{tag_suffix(tag)}_{fold}.model"
    )

    last_epoch = start_epoch
    for epoch in range(start_epoch, NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        print("Predicting for validation data...")
        G, P = predicting(model, device, valid_loader)
        val_mse = get_mse(G, P)
        print(f"Epoch {epoch + 1} - Validation MSE: {val_mse}, Best MSE: {best_mse}")
        improved = val_mse < best_mse
        if improved:
            best_mse = val_mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            print(f"MSE improved at epoch {best_epoch}; Best MSE: {best_mse}")
        else:
            print(f"No improvement since epoch {best_epoch}; Best MSE: {best_mse}")

        if improved or (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0:
            save_checkpoint(
                checkpoint_file_name,
                fold,
                epoch + 1,
                model,
                optimizer,
                best_mse,
                best_epoch,
            )

        last_epoch = epoch + 1
        if patience_exhausted(last_epoch, best_epoch):
            print(
                f"Early stopping fold {fold} at epoch {last_epoch}: no improvement for "
                f"{last_epoch - best_epoch} epochs (best MSE {best_mse} at epoch {best_epoch})"
            )
            break

    record_best_epoch(paths.results, fold, best_epoch, best_mse, last_epoch, tag)


def check_full_split(n_rows: int, paths: DatasetPaths) -> None:
    train_idx = {i for fold in load_fold_indices(paths.train_folds) for i in fold}
    assert n_rows == len(train_idx) == FULL_TRAIN_ROWS, (
        f"Full training set has {n_rows} rows ({len(train_idx)} unique indices), "
        f"expected {FULL_TRAIN_ROWS}"
    )
    assert train_idx.isdisjoint(load_test_indices(paths.test_fold)), (
        "Full training set shares indices with the test fold"
    )


def run_full(
    epochs: int,
    condition: Condition,
    tag: str,
    seed: int,
    paths: DatasetPaths,
    device: torch.device,
) -> None:
    set_seed(seed)
    model = condition.build_model()
    model.to(device)
    model_st = condition.model_name
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    checkpoint_file_name = checkpoint_path(
        paths.models, model_st, dataset, FULL_FOLD, tag
    )
    start_epoch, _, _ = load_checkpoint(
        checkpoint_file_name, FULL_FOLD, model, optimizer, device
    )
    if start_epoch > epochs:
        raise SystemExit(
            f"Checkpoint {checkpoint_file_name} is at epoch {start_epoch}, "
            f"past the requested {epochs} epochs"
        )
    if start_epoch == epochs:
        print(f"Full-data training already finished at epoch {start_epoch}")
    elif start_epoch > 0:
        print(f"Resuming full-data training at epoch {start_epoch + 1}")

    print(f"Training on all training folds for {epochs} epochs...")

    train_data = condition.full_train_dataset()
    check_full_split(len(train_data), paths)
    train_loader = make_train_loader(train_data, seed, condition.collate)

    for epoch in range(start_epoch, epochs):
        train(model, device, train_loader, optimizer, epoch + 1)
        if (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0:
            save_checkpoint(
                checkpoint_file_name,
                FULL_FOLD,
                epoch + 1,
                model,
                optimizer,
                float("inf"),
                -1,
            )

    model_file_name = os.path.join(
        paths.models, f"model_{model_st}_{dataset}{tag_suffix(tag)}_full.model"
    )
    torch.save(model.state_dict(), model_file_name)
    print(f"Saved full-data model after {epochs} epochs to {model_file_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train GNNNet or PLMNet on {dataset}."
    )
    parser.add_argument(
        "--mode",
        choices=["cv", "full"],
        default="cv",
        help="cv: 5-fold cross-validation with early stopping. "
        "full: train on all five training folds with no validation.",
    )
    parser.add_argument(
        "--epochs", type=int, help="Exact number of epochs, required for --mode full."
    )
    add_run_args(parser)
    args = parser.parse_args()
    check_run_args(parser, args)
    if args.mode == "full" and args.epochs is None:
        parser.error("--epochs is required with --mode full")
    if args.mode == "cv" and args.epochs is not None:
        parser.error("--epochs only applies to --mode full")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    return args


def main() -> None:
    args = parse_args()

    print("Dataset: ", dataset)
    print("Mode: ", args.mode)
    print("Protein representation: ", args.protein_repr)
    print("Structure set: ", args.pdb_set)
    if args.plm_model is not None:
        print("Protein language model: ", args.plm_model)
        print("Standardized embeddings: ", args.standardize)
    print("Seed: ", args.seed)
    print("Learning rate: ", LR)
    if args.mode == "cv":
        print("Epochs: ", NUM_EPOCHS)
        print("Early stop patience: ", EARLY_STOP_PATIENCE)
    else:
        print("Epochs: ", args.epochs)

    paths = paths_for(dataset, args.pdb_set)
    print("Data root: ", paths.root)
    os.makedirs(paths.models, exist_ok=True)
    os.makedirs(paths.results, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tag = run_tag(
        args.protein_repr, args.plm_model, args.seed, args.standardize, args.pdb_set
    )
    print("Run tag: ", tag or "(none)")
    condition = build_condition(
        args.protein_repr, args.plm_model, args.standardize, dataset, paths
    )

    if args.mode == "full":
        run_full(args.epochs, condition, tag, args.seed, paths, device)
        return

    for fold in range(len(load_fold_indices(paths.train_folds))):
        run_fold(fold, condition, tag, args.seed, paths, device)
        gc.collect()


if __name__ == "__main__":
    main()
