import os
from dataclasses import dataclass

ROOT_ENV_VAR = "DTA_DATA_ROOT"

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def data_root() -> str:
    return os.environ.get(ROOT_ENV_VAR) or _REPO_DIR


@dataclass(frozen=True)
class DatasetPaths:
    root: str
    dataset_dir: str
    csv: str
    train_folds: str
    test_fold: str
    pdb: str
    contact_maps: str
    ligand_graphs: str
    protein_graphs: str
    protein_graph_gml: str
    models: str
    results: str


def paths_for(dataset: str = "davis") -> DatasetPaths:
    root = data_root()
    dataset_dir = os.path.join(root, "data", dataset)
    protein_graphs = os.path.join(dataset_dir, "protein_graphs")
    return DatasetPaths(
        root=root,
        dataset_dir=dataset_dir,
        csv=os.path.join(dataset_dir, f"{dataset}.csv"),
        train_folds=os.path.join(dataset_dir, "train_folds.txt"),
        test_fold=os.path.join(dataset_dir, "test_fold.txt"),
        pdb=os.path.join(dataset_dir, "pdb"),
        contact_maps=os.path.join(dataset_dir, "contact_maps"),
        ligand_graphs=os.path.join(dataset_dir, "ligand_graphs"),
        protein_graphs=protein_graphs,
        protein_graph_gml=os.path.join(protein_graphs, "gml"),
        models=os.path.join(root, f"models_{dataset}"),
        results=os.path.join(root, f"results_{dataset}"),
    )
