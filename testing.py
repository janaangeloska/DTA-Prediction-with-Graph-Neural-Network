import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from Creating_Train_and_Test_set import create_test_dataset
from DTADataset import *
from Emetrics import (
    get_ci,
    get_cindex,
    get_mse,
    get_pearson,
    get_rm2,
    get_rmse,
    get_spearman,
)
from GNNNet import GNNNet
from Paths import paths_for


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


def format_report(per_fold, dataset):
    folds = sorted(per_fold)
    lines = [f"{dataset}  folds evaluated: {folds}", ""]
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


def plot_density(Y, P, dataset, results_path, fold):
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
        os.path.join(results_path, f"{dataset}_fold{fold}.png"),
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    dataset = "davis"
    cuda_name = "cuda:0"
    TEST_BATCH_SIZE = 512
    FOLDS = [0, 1, 2, 3, 4]

    model_st = GNNNet.__name__
    PATHS = paths_for(dataset)
    results_path = PATHS.results
    os.makedirs(results_path, exist_ok=True)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    print(f"dataset: {dataset}")
    print(f"device: {device}")

    test_data = create_test_dataset(dataset)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    per_fold = {}
    for fold in FOLDS:
        model_file_name = os.path.join(
            PATHS.models, f"model_{model_st}_{dataset}_{fold}.model"
        )
        if not os.path.exists(model_file_name):
            print(f"Fold {fold}: no model at {model_file_name}, skipping.")
            continue

        model = GNNNet().to(device)
        model.load_state_dict(torch.load(model_file_name, map_location=device))
        Y, P = predicting(model, device, test_loader)

        per_fold[fold] = calculate_metrics(Y, P)
        print(f"Fold {fold} metrics:")
        for name in METRIC_NAMES:
            print(f"  {name}: {per_fold[fold][name]}")
        plot_density(Y, P, dataset, results_path, fold)

    if not per_fold:
        raise SystemExit("No fold models found, nothing to evaluate.")

    report = format_report(per_fold, dataset)
    print()
    print(report)
    result_file_name = os.path.join(results_path, f"result_{model_st}_{dataset}.txt")
    with open(result_file_name, "w") as file:
        file.write(report + "\n")
    print(f"\nWrote {result_file_name}")
