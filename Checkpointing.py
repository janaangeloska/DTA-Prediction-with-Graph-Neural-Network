import os
from typing import Tuple

import torch
from torch import nn


def checkpoint_path(models_dir: str, model_name: str, dataset: str, fold: int) -> str:
    return os.path.join(models_dir, f"checkpoint_{model_name}_{dataset}_{fold}.pt")


def save_checkpoint(
    path: str,
    fold: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_mse: float,
    best_epoch: int,
) -> None:
    payload = {
        "fold": fold,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mse": best_mse,
        "best_epoch": best_epoch,
    }
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    fold: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, float, int]:
    if not os.path.exists(path):
        return 0, float("inf"), -1

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["fold"] != fold:
        raise ValueError(
            f"Checkpoint {path} holds fold {checkpoint['fold']}, expected fold {fold}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_mse"], checkpoint["best_epoch"]
