from collections.abc import Callable
from dataclasses import dataclass

from torch import nn
from torch.utils.data import Dataset

from Creating_Train_and_Test_set import (
    create_dataset_for_5folds,
    create_plm_dataset_for_folds,
    create_plm_test_dataset,
    create_test_dataset,
)
from DTADataset import collate, plm_collate
from GNNNet import GNNNet
from Paths import DatasetPaths
from PLMNet import PLMNet
from Protein_Embeddings import load_pooled_embeddings
from RunTag import MODEL_CLASS_NAMES


@dataclass(frozen=True)
class Condition:
    """Everything that differs between protein representations: model, data, batching."""

    model_name: str
    build_model: Callable[[], nn.Module]
    fold_datasets: Callable[[int], tuple[Dataset, Dataset]]
    full_train_dataset: Callable[[], Dataset]
    test_dataset: Callable[[], Dataset]
    collate: Callable


def structure_condition(dataset: str, paths: DatasetPaths) -> Condition:
    return Condition(
        model_name=MODEL_CLASS_NAMES["structure"],
        build_model=GNNNet,
        fold_datasets=lambda fold: create_dataset_for_5folds(
            dataset_name=dataset, paths=paths, fold_idx=fold
        ),
        full_train_dataset=lambda: create_dataset_for_5folds(
            dataset_name=dataset, paths=paths, combine_all=True
        )[0],
        test_dataset=lambda: create_test_dataset(dataset, paths),
        collate=collate,
    )


def plm_condition(paths: DatasetPaths, plm_model: str, standardize: bool) -> Condition:
    embeddings = load_pooled_embeddings(
        paths.protein_embeddings, plm_model, paths.pdb_file, standardize=standardize
    )
    embedding_dim = next(iter(embeddings.values())).shape[0]
    print(
        f"Loaded {plm_model} embeddings for {len(embeddings)} proteins, dim {embedding_dim}"
    )
    return Condition(
        model_name=MODEL_CLASS_NAMES["plm"],
        build_model=lambda: PLMNet(embedding_dim=embedding_dim),
        fold_datasets=lambda fold: create_plm_dataset_for_folds(
            embeddings, paths, fold_idx=fold
        ),
        full_train_dataset=lambda: create_plm_dataset_for_folds(
            embeddings, paths, combine_all=True
        )[0],
        test_dataset=lambda: create_plm_test_dataset(embeddings, paths),
        collate=plm_collate,
    )


def build_condition(
    protein_repr: str,
    plm_model: str | None,
    standardize: bool,
    dataset: str,
    paths: DatasetPaths,
) -> Condition:
    if protein_repr == "plm":
        return plm_condition(paths, plm_model, standardize)
    return structure_condition(dataset, paths)
