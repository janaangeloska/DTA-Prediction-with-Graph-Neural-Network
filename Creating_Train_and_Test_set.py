import glob
import hashlib
import json
import os
import re

import networkx as nx
import numpy as np
import pandas as pd

from DTADataset import DTADataset
from Paths import paths_for

PATHS = paths_for("davis")
ligand_folder = PATHS.ligand_graphs
protein_folder = PATHS.protein_graph_gml
csv_file = PATHS.csv
train_fold_file = PATHS.train_folds
test_fold_file = PATHS.test_fold


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


ligand_files = glob.glob(os.path.join(ligand_folder, "*.gml"))
protein_files = glob.glob(os.path.join(protein_folder, "*.gml"))

smile_graph = {}
target_graph = {}


for ligand_file in ligand_files:
    base_name = os.path.basename(ligand_file)
    ligand_name = base_name.replace(".gml", "")
    smile_graph[ligand_name] = load_ligand_gml(ligand_file)


for protein_file in protein_files:
    base_name = os.path.basename(protein_file)
    protein_name = sanitize_protein_filename(base_name)
    target_graph[protein_name] = load_protein_gml(protein_file)


data_df = pd.read_csv(csv_file)
data_df["ligands"] = data_df["ligand"].apply(sanitize_filename_DAVIS)

ligands = data_df["ligands"].tolist()
proteins = data_df["protein"].apply(sanitize_protein_filename).tolist()
labels = data_df["label"].tolist()


def load_fold_indices(train_fold_file, num_folds=5):
    with open(train_fold_file, "r") as f:
        fold_indices = json.load(f)
    assert len(fold_indices) == num_folds, (
        f"Expected {num_folds} folds, got {len(fold_indices)}"
    )
    return fold_indices


fold_indices = load_fold_indices(train_fold_file)
print(f"Loaded fold indices (number of folds): {len(fold_indices)}")


def create_dataset_for_5folds(dataset_name, combine_all=False, fold_idx=0):
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

    train_ligands = np.array(ligands)[train_idx]
    train_proteins = np.array(proteins)[train_idx]
    train_labels = np.array(labels)[train_idx]

    val_ligands, val_proteins, val_labels = [], [], []
    if not combine_all:  # Only for specific fold validation
        val_ligands = np.array(ligands)[val_idx]
        val_proteins = np.array(proteins)[val_idx]
        val_labels = np.array(labels)[val_idx]

    train_dataset = DTADataset(
        root="/tmp",
        dataset=dataset_name + "_train",
        xd=train_ligands.tolist(),
        y=train_labels.tolist(),
        smile_graph=smile_graph,
        target_key=train_proteins.tolist(),
        target_graph=target_graph,
    )

    val_dataset = None
    if not combine_all:
        # Validation dataset
        val_dataset = DTADataset(
            root="/tmp",
            dataset=dataset_name + "_valid",
            xd=val_ligands.tolist(),
            y=val_labels.tolist(),
            smile_graph=smile_graph,
            target_key=val_proteins.tolist(),
            target_graph=target_graph,
        )

    return train_dataset, val_dataset


def load_test_indices(test_fold_file):
    with open(test_fold_file, "r") as f:
        content = f.read().strip()
        # Parse test indices
        content = content.replace("[", "").replace("]", "").replace(" ", "")
        test_indices = list(map(int, content.split(",")))
    return test_indices


test_indices = load_test_indices(test_fold_file)


def create_test_dataset(dataset_name):
    test_ligands = np.array(ligands)[test_indices]
    test_proteins = np.array(proteins)[test_indices]
    test_labels = np.array(labels)[test_indices]

    test_dataset = DTADataset(
        root="/tmp",
        dataset=dataset_name + "_test",
        xd=test_ligands.tolist(),
        y=test_labels.tolist(),
        smile_graph=smile_graph,
        target_key=test_proteins.tolist(),
        target_graph=target_graph,
    )

    print(f"Test dataset created with {len(test_indices)} samples.")
    return test_dataset
