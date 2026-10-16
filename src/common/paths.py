import os
from collections.abc import Mapping
from dataclasses import dataclass

from src.common.split_sets import DEFAULT_SPLIT, SPLITS
from src.common.structure_sets import DEFAULT_STRUCTURE_SET, STRUCTURE_SETS

ROOT_ENV_VAR = "DTA_DATA_ROOT"

# This file sits in src/common, two levels below the repo root.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def data_root() -> str:
    return os.environ.get(ROOT_ENV_VAR) or _REPO_DIR


@dataclass(frozen=True)
class DatasetPaths:
    root: str
    dataset_dir: str
    csv: str
    splits: str
    train_folds: str
    test_fold: str
    pdb_set: str
    pdb: str
    pdb_file_stems: Mapping[str, str]
    contact_maps: str
    ligand_graphs: str
    protein_graphs: str
    protein_graph_gml: str
    protein_embeddings: str
    models: str
    results: str

    def pdb_file(self, name: str) -> str:
        return os.path.join(self.pdb, f"{self.pdb_file_stems.get(name, name)}.pdb")


def paths_for(
    dataset: str = "davis",
    pdb_set: str = DEFAULT_STRUCTURE_SET,
    split: str = DEFAULT_SPLIT,
) -> DatasetPaths:
    root = data_root()
    dataset_dir = os.path.join(root, "data", dataset)
    splits = os.path.join(dataset_dir, "splits")
    fold_dir = dataset_dir
    if SPLITS[split].dir_name:
        fold_dir = os.path.join(splits, SPLITS[split].dir_name)
    structure_set = STRUCTURE_SETS[pdb_set]
    suffix = structure_set.dir_suffix
    protein_graphs = os.path.join(dataset_dir, f"protein_graphs{suffix}")
    return DatasetPaths(
        root=root,
        dataset_dir=dataset_dir,
        csv=os.path.join(dataset_dir, f"{dataset}.csv"),
        splits=splits,
        train_folds=os.path.join(fold_dir, "train_folds.txt"),
        test_fold=os.path.join(fold_dir, "test_fold.txt"),
        pdb_set=pdb_set,
        pdb=os.path.join(dataset_dir, f"pdb{suffix}"),
        pdb_file_stems=structure_set.file_stems,
        contact_maps=os.path.join(dataset_dir, f"contact_maps{suffix}"),
        ligand_graphs=os.path.join(dataset_dir, "ligand_graphs"),
        protein_graphs=protein_graphs,
        protein_graph_gml=os.path.join(protein_graphs, "gml"),
        protein_embeddings=os.path.join(dataset_dir, f"protein_embeddings{suffix}"),
        models=os.path.join(root, f"models_{dataset}"),
        results=os.path.join(root, f"results_{dataset}"),
    )
