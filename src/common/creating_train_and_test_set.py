import functools
import glob
import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.common.dta_dataset import DTADataset, PLMDTADataset
from src.common.paths import DatasetPaths


def sanitize_filename_KIBA(smile):
    smile_hash = hashlib.md5(smile.encode()).hexdigest()
    return smile_hash


def sanitize_filename_DAVIS(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def sanitize_protein_filename(filename):
    return re.sub(r"_contact_map_graph\.gml$", "", filename)


def load_ligand_gml(file_path):
    graph = nx.read_gml(file_path)
    features = []
    edge_index = []
    for node, data in graph.nodes(data=True):
        features.append(list(map(float, data["feature"].split(","))))
    for source, target in graph.edges:
        edge_index.append([int(source), int(target)])
        edge_index.append([int(target), int(source)])
    return len(features), features, edge_index


def load_protein_gml(file_path):
    graph = nx.read_gml(file_path)
    features = []
    edge_index = []
    for node, data in graph.nodes(data=True):
        features.append(list(map(float, data["features"][1:-1].split(","))))
    for source, target in graph.edges:
        edge_index.append([int(source), int(target)])
        edge_index.append([int(target), int(source)])
    return len(features), features, edge_index


# Cached per folder, so the five folds and the test set reuse one parse of the graph files.
@functools.cache
def ligand_graphs(ligand_folder: str) -> dict[str, tuple[int, list, list]]:
    graphs = {}
    for ligand_file in glob.glob(os.path.join(ligand_folder, "*.gml")):
        ligand_name = os.path.basename(ligand_file).replace(".gml", "")
        graphs[ligand_name] = load_ligand_gml(ligand_file)
    return graphs


@functools.cache
def protein_graphs(protein_folder: str) -> dict[str, tuple[int, list, list]]:
    graphs = {}
    for protein_file in glob.glob(os.path.join(protein_folder, "*.gml")):
        protein_name = sanitize_protein_filename(os.path.basename(protein_file))
        graphs[protein_name] = load_protein_gml(protein_file)
    if not graphs:
        raise FileNotFoundError(
            f"No protein graphs in {protein_folder}. Run "
            "python -m src.structure.protein_representation with the same --pdb-set first."
        )
    return graphs


@dataclass(frozen=True)
class PairTable:
    ligands: list[str]
    proteins: list[str]
    labels: list[float]


@functools.cache
def pair_table(csv_file: str) -> PairTable:
    data_df = pd.read_csv(csv_file)
    return PairTable(
        ligands=data_df["ligand"].apply(sanitize_filename_DAVIS).tolist(),
        proteins=data_df["protein"].apply(sanitize_protein_filename).tolist(),
        labels=data_df["label"].tolist(),
    )


def load_fold_indices(train_fold_file, num_folds=5):
    with open(train_fold_file, "r") as f:
        fold_indices = json.load(f)
    assert len(fold_indices) == num_folds, (
        f"Expected {num_folds} folds, got {len(fold_indices)}"
    )
    return fold_indices


def split_indices(
    train_fold_file: str, combine_all: bool = False, fold_idx: int = 0
) -> tuple[list[int], list[int]]:
    fold_indices = load_fold_indices(train_fold_file)

    if combine_all:
        train_idx = [idx for fold in fold_indices for idx in fold]
        val_idx = []  # No validation set in this case
    else:
        if fold_idx >= len(fold_indices):
            raise ValueError(
                f"Fold index {fold_idx} out of range. There are only {len(fold_indices)} folds."
            )

        val_idx = fold_indices[fold_idx]
        # Train on the other folds only. Ranging over the full dataset would pull the
        # held-out test rows, which are absent from the fold file, into every fold.
        train_idx = [
            i for n, fold in enumerate(fold_indices) if n != fold_idx for i in fold
        ]
    return train_idx, val_idx


def create_structure_dataset(
    dataset_name: str, indices: Sequence[int], paths: DatasetPaths
) -> DTADataset:
    table = pair_table(paths.csv)
    return DTADataset(
        root="/tmp",
        dataset=dataset_name,
        xd=np.array(table.ligands)[indices].tolist(),
        y=np.array(table.labels)[indices].tolist(),
        smile_graph=ligand_graphs(paths.ligand_graphs),
        target_key=np.array(table.proteins)[indices].tolist(),
        target_graph=protein_graphs(paths.protein_graph_gml),
    )


def create_dataset_for_5folds(
    dataset_name: str,
    paths: DatasetPaths,
    combine_all: bool = False,
    fold_idx: int = 0,
) -> tuple[DTADataset, DTADataset | None]:
    train_idx, val_idx = split_indices(paths.train_folds, combine_all, fold_idx)
    train_dataset = create_structure_dataset(dataset_name + "_train", train_idx, paths)
    val_dataset = None
    if not combine_all:
        val_dataset = create_structure_dataset(dataset_name + "_valid", val_idx, paths)
    return train_dataset, val_dataset


def load_test_indices(test_fold_file):
    with open(test_fold_file, "r") as f:
        content = f.read().strip()
        # Parse test indices
        content = content.replace("[", "").replace("]", "").replace(" ", "")
        test_indices = list(map(int, content.split(",")))
    return test_indices


def create_test_dataset(dataset_name: str, paths: DatasetPaths) -> DTADataset:
    test_indices = load_test_indices(paths.test_fold)
    test_dataset = create_structure_dataset(dataset_name + "_test", test_indices, paths)
    print(f"Test dataset created with {len(test_indices)} samples.")
    return test_dataset


def create_plm_dataset(
    indices: Sequence[int],
    target_embedding: dict[str, torch.Tensor],
    paths: DatasetPaths,
) -> PLMDTADataset:
    table = pair_table(paths.csv)
    return PLMDTADataset(
        xd=[table.ligands[i] for i in indices],
        y=[table.labels[i] for i in indices],
        smile_graph=ligand_graphs(paths.ligand_graphs),
        target_key=[table.proteins[i] for i in indices],
        target_embedding=target_embedding,
    )


def create_plm_dataset_for_folds(
    target_embedding: dict[str, torch.Tensor],
    paths: DatasetPaths,
    combine_all: bool = False,
    fold_idx: int = 0,
) -> tuple[PLMDTADataset, PLMDTADataset | None]:
    train_idx, val_idx = split_indices(paths.train_folds, combine_all, fold_idx)
    train_dataset = create_plm_dataset(train_idx, target_embedding, paths)
    val_dataset = (
        None if combine_all else create_plm_dataset(val_idx, target_embedding, paths)
    )
    return train_dataset, val_dataset


def create_plm_test_dataset(
    target_embedding: dict[str, torch.Tensor], paths: DatasetPaths
) -> PLMDTADataset:
    return create_plm_dataset(
        load_test_indices(paths.test_fold), target_embedding, paths
    )
