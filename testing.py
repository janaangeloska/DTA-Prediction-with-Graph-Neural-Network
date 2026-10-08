import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.conditions import build_condition
from src.common.creating_train_and_test_set import load_fold_indices
from src.common.dta_dataset import *
from src.common.emetrics import (
    get_ci,
    get_cindex,
    get_mse,
    get_pearson,
    get_rm2,
    get_rmse,
    get_spearman,
)
from src.common.paths import paths_for
from src.common.run_tag import add_run_args, check_run_args, run_tag, tag_suffix
from src.common.seeding import set_seed


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print(f"Make prediction for {len(loader.dataset)} samples...")
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


METRIC_NAMES = ["rmse", "mse", "pearson", "spearman", "ci", "cindex", "rm2"]


def calculate_metrics(Y, P):
    return {
        "rmse": get_rmse(Y, P),
        "mse": get_mse(Y, P),
        "pearson": get_pearson(Y, P),
        "spearman": get_spearman(Y, P),
        "ci": get_ci(Y, P),
        "cindex": get_cindex(Y, P),
        "rm2": get_rm2(Y, P),
    }


def format_report(per_fold: dict, dataset: str) -> str:
    folds = sorted(per_fold)
    lines = [
        f"{dataset}  folds evaluated: {folds}",
        (
            "ci (all pairs) is the value to compare with published numbers; cindex "
            "only counts pairs whose larger label has the larger row index."
        ),
        "",
    ]
    for fold in folds:
        row = "  ".join(f"{n}: {per_fold[fold][n]:.4f}" for n in METRIC_NAMES)
        lines.append(f"fold {fold}  {row}")
    lines.append("")
    for name in METRIC_NAMES:
        values = np.array([per_fold[f][name] for f in folds], dtype=float)
        # Sample std across folds; undefined for a single fold.
        std = values.std(ddof=1) if len(values) > 1 else float("nan")
        lines.append(f"{name:9s} mean: {values.mean():.4f}  std: {std:.4f}")
    return "\n".join(lines)


def plot_density(Y, P, dataset, results_path, fold, suffix=""):
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle="--")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.scatter(P, Y, color="blue", s=40)
    plt.title(f"density of {dataset}", fontsize=30, fontweight="bold")
    plt.xlabel("predicted", fontsize=30, fontweight="bold")
    plt.ylabel("measured", fontsize=30, fontweight="bold")
    plt.plot([5, 11], [5, 11], color="black")
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight="bold")
    plt.savefig(
        os.path.join(results_path, f"{dataset}{suffix}_fold{fold}.png"),
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score trained GNNNet or PLMNet models on the test set."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Score the full-data model instead of the five fold models.",
    )
    add_run_args(parser)
    args = parser.parse_args()
    check_run_args(parser, args)

    set_seed(args.seed)
    dataset = "davis"
    cuda_name = "cuda:0"
    TEST_BATCH_SIZE = 512

    PATHS = paths_for(dataset, args.pdb_set)
    condition = build_condition(
        args.protein_repr, args.plm_model, args.standardize, dataset, PATHS
    )
    model_st = condition.model_name
    suffix = tag_suffix(
        run_tag(
            args.protein_repr, args.plm_model, args.seed, args.standardize, args.pdb_set
        )
    )
    results_path = PATHS.results
    os.makedirs(results_path, exist_ok=True)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    print(f"dataset: {dataset}")
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"structure set: {args.pdb_set}")

    test_data = condition.test_dataset()
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        collate_fn=condition.collate,
    )

    if args.full:
        fold_labels = ["full"]
        result_suffix = "_full"
    else:
        fold_labels = list(range(len(load_fold_indices(PATHS.train_folds))))
        result_suffix = ""

    per_fold = {}
    for fold in fold_labels:
        model_file_name = os.path.join(
            PATHS.models, f"model_{model_st}_{dataset}{suffix}_{fold}.model"
        )
        if not os.path.exists(model_file_name):
            print(f"Fold {fold}: no model at {model_file_name}, skipping.")
            continue

        model = condition.build_model().to(device)
        model.load_state_dict(torch.load(model_file_name, map_location=device))
        Y, P = predicting(model, device, test_loader)

        per_fold[fold] = calculate_metrics(Y, P)
        print(f"Fold {fold} metrics:")
        for name in METRIC_NAMES:
            print(f"  {name}: {per_fold[fold][name]}")
        plot_density(Y, P, dataset, results_path, fold, suffix)

    if not per_fold:
        raise SystemExit("No fold models found, nothing to evaluate.")

    report = format_report(per_fold, dataset)
    print()
    print(report)
    result_file_name = os.path.join(
        results_path, f"result_{model_st}_{dataset}{suffix}{result_suffix}.txt"
    )
    with open(result_file_name, "w") as file:
        file.write(report + "\n")
    print(f"\nWrote {result_file_name}")
