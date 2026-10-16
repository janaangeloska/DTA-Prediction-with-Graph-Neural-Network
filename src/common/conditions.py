from collections.abc import Callable
from dataclasses import dataclass

from torch import nn
from torch.utils.data import Dataset

from src.common.creating_train_and_test_set import (
    create_dataset_for_5folds,
    create_plm_dataset_for_folds,
    create_plm_test_dataset,
    create_test_dataset,
    protein_graphs,
    residue_alignment_mismatches,
)
from src.common.dta_dataset import collate, plm_collate
from src.common.paths import DatasetPaths
from src.common.run_tag import HYBRID_REPRS, MODEL_CLASS_NAMES
from src.common.seeding import text_seed
from src.hybrid.hybrid_net import HybridNet
from src.sequence.plm_net import PLMNet
from src.sequence.protein_embeddings import (
    build_adapter,
    load_pooled_embeddings,
    load_residue_embeddings,
)
from src.structure.gnn_net import GNNNet
from src.structure.protein_representation import rewired_edges, self_loop_edges

HYBRID_EDGES: dict[str, Callable[[str, int, list], list]] = {
    "hybrid": lambda name, size, edge_index: edge_index,
    "hybrid_noedges": lambda name, size, edge_index: self_loop_edges(size),
    "hybrid_shuffled": lambda name, size, edge_index: rewired_edges(
        size, edge_index, text_seed(name)
    ),
}


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


def hybrid_condition(
    dataset: str, paths: DatasetPaths, protein_repr: str, plm_model: str
) -> Condition:
    residues = load_residue_embeddings(paths.protein_embeddings, plm_model, paths.pdb_file)
    graphs = protein_graphs(paths.protein_graph_gml)
    mismatches = residue_alignment_mismatches(graphs, residues)
    if mismatches:
        listing = ", ".join(
            f"{name} ({nodes} nodes, {length} residues)"
            for name, (nodes, length) in sorted(mismatches.items())
        )
        raise ValueError(
            f"{len(mismatches)} proteins have graphs and {plm_model} residue embeddings "
            f"of different lengths: {listing}"
        )
    make_edges = HYBRID_EDGES[protein_repr]
    target_graph = {
        name: (size, features, make_edges(name, size, edge_index))
        for name, (size, features, edge_index) in graphs.items()
    }
    embedding_dim = build_adapter(plm_model).embedding_dim
    print(
        f"Loaded per-residue {plm_model} embeddings for {len(residues)} proteins, "
        f"dim {embedding_dim}, all aligned with the graph nodes"
    )
    return Condition(
        model_name=MODEL_CLASS_NAMES[protein_repr],
        build_model=lambda: HybridNet(embedding_dim=embedding_dim),
        fold_datasets=lambda fold: create_dataset_for_5folds(
            dataset_name=dataset,
            paths=paths,
            fold_idx=fold,
            target_graph=target_graph,
            target_residues=residues,
        ),
        full_train_dataset=lambda: create_dataset_for_5folds(
            dataset_name=dataset,
            paths=paths,
            combine_all=True,
            target_graph=target_graph,
            target_residues=residues,
        )[0],
        test_dataset=lambda: create_test_dataset(dataset, paths, target_graph, residues),
        collate=collate,
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
    if protein_repr in HYBRID_REPRS:
        return hybrid_condition(dataset, paths, protein_repr, plm_model)
    return structure_condition(dataset, paths)
