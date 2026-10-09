import argparse
import itertools
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.common.paths import DatasetPaths, paths_for
from src.common.split_sets import COLD_SPLITS
from src.structure.protein_representation import pdb_sequences

DATASET = "davis"
SEQUENCE_PDB_SET = "kinase_domain"
COLD_PROTEIN_THRESHOLD = 0.95
FAMILY_THRESHOLD_CANDIDATES = (0.30, 0.35, 0.40, 0.45, 0.50)
REPORTED_THRESHOLDS = (0.95, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30)
MAX_CLUSTER_FRACTION = 0.25
TEST_FRACTION = 0.20
TEST_FRACTION_TOLERANCE = 0.03
N_FOLDS = 5
SPLIT_SEED = 0


def build_aligner() -> PairwiseAligner:
    return PairwiseAligner(
        mode="global",
        substitution_matrix=substitution_matrices.load("BLOSUM62"),
        open_gap_score=-10,
        extend_gap_score=-0.5,
        end_gap_score=0,
    )


def pairwise_identity(a: str, b: str, aligner: PairwiseAligner) -> float:
    alignment = aligner.align(a, b)[0]
    return alignment.counts().identities / min(len(a), len(b))


def identity_matrix(sequences: list[str]) -> np.ndarray:
    aligner = build_aligner()
    identity = np.eye(len(sequences))
    for i, j in itertools.combinations(range(len(sequences)), 2):
        identity[i, j] = identity[j, i] = pairwise_identity(
            sequences[i], sequences[j], aligner
        )
    return identity


def cached_identity_matrix(cache_dir: str, sequences: list[str]) -> np.ndarray:
    matrix_file = os.path.join(cache_dir, f"identity_{SEQUENCE_PDB_SET}.npy")
    order_file = os.path.join(cache_dir, f"identity_{SEQUENCE_PDB_SET}_sequences.json")
    if os.path.exists(matrix_file) and os.path.exists(order_file):
        with open(order_file) as file:
            if json.load(file) == sequences:
                return np.load(matrix_file)
    print(f"Aligning {len(sequences)} unique sequences pairwise...")
    identity = identity_matrix(sequences)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(matrix_file, identity)
    with open(order_file, "w") as file:
        json.dump(sequences, file)
    return identity


def protein_clusters(
    sequence_of: dict[str, str],
    sequences: list[str],
    identity: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    _, labels = connected_components(csr_matrix(identity >= threshold), directed=False)
    index = {sequence: i for i, sequence in enumerate(sequences)}
    return {name: int(labels[index[seq]]) for name, seq in sequence_of.items()}


def cluster_stats(cluster_of: dict[str, int]) -> tuple[int, int, int]:
    sizes = Counter(cluster_of.values())
    singletons = sum(1 for size in sizes.values() if size == 1)
    return len(sizes), max(sizes.values()), singletons


def print_threshold_table(
    sequence_of: dict[str, str], sequences: list[str], identity: np.ndarray
) -> dict[float, int]:
    print(f"{'threshold':>9}  {'clusters':>8}  {'largest':>7}  {'singletons':>10}")
    largest_at = {}
    for threshold in REPORTED_THRESHOLDS:
        cluster_of = protein_clusters(sequence_of, sequences, identity, threshold)
        n_clusters, largest, singletons = cluster_stats(cluster_of)
        largest_at[threshold] = largest
        print(f"{threshold:>9.2f}  {n_clusters:>8}  {largest:>7}  {singletons:>10}")
    return largest_at


def choose_family_threshold(largest_at: dict[float, int], n_proteins: int) -> float:
    for threshold in sorted(FAMILY_THRESHOLD_CANDIDATES):
        if largest_at[threshold] <= MAX_CLUSTER_FRACTION * n_proteins:
            return threshold
    raise SystemExit(
        f"No threshold in {FAMILY_THRESHOLD_CANDIDATES} keeps the largest cluster at "
        f"or below {MAX_CLUSTER_FRACTION:.0%} of {n_proteins} proteins"
    )


def choose_test_clusters(
    cluster_sizes: dict[int, int], target: float, rng: np.random.Generator
) -> set[int]:
    chosen, total = set(), 0
    for cluster in rng.permutation(sorted(cluster_sizes)):
        size = cluster_sizes[int(cluster)]
        if abs(total + size - target) < abs(total - target):
            chosen.add(int(cluster))
            total += size
    return chosen


def assign_folds(
    cluster_rows: dict[int, int], rng: np.random.Generator
) -> list[list[int]]:
    order = sorted(
        (int(c) for c in rng.permutation(sorted(cluster_rows))),
        key=lambda c: -cluster_rows[c],
    )
    folds = [[] for _ in range(N_FOLDS)]
    rows = [0] * N_FOLDS
    for cluster in order:
        lightest = rows.index(min(rows))
        folds[lightest].append(cluster)
        rows[lightest] += cluster_rows[cluster]
    if not all(folds):
        raise SystemExit(f"Fewer than {N_FOLDS} training clusters, a fold is empty")
    return folds


def build_split(
    cluster_of: dict[str, int], protein_rows: dict[str, list[int]], seed: int
) -> tuple[list[list[int]], list[int], set[int], list[set[int]]]:
    rng = np.random.default_rng(seed)
    cluster_sizes = Counter(cluster_of.values())
    test_clusters = choose_test_clusters(
        cluster_sizes, TEST_FRACTION * len(cluster_of), rng
    )
    cluster_rows = Counter()
    for name, cluster in cluster_of.items():
        if cluster not in test_clusters:
            cluster_rows[cluster] += len(protein_rows[name])
    fold_clusters = [set(fold) for fold in assign_folds(cluster_rows, rng)]

    def rows_in(clusters: set[int]) -> list[int]:
        return sorted(
            row
            for name, cluster in cluster_of.items()
            if cluster in clusters
            for row in protein_rows[name]
        )

    train_folds = [rows_in(clusters) for clusters in fold_clusters]
    return train_folds, rows_in(test_clusters), test_clusters, fold_clusters


def write_frozen(path: str, text: str) -> None:
    if os.path.exists(path):
        with open(path) as file:
            if file.read() != text:
                raise SystemExit(
                    f"{path} exists with different content. Delete its folder to rebuild."
                )
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write(text)


def write_split(
    paths: DatasetPaths,
    train_folds: list[list[int]],
    test_fold: list[int],
    info: dict,
) -> None:
    write_frozen(paths.train_folds, json.dumps(train_folds))
    write_frozen(paths.test_fold, json.dumps(test_fold))
    info_file = os.path.join(os.path.dirname(paths.train_folds), "split_info.json")
    write_frozen(info_file, json.dumps(info, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Build a {DATASET} split that holds out whole protein clusters."
    )
    parser.add_argument("--split", choices=COLD_SPLITS, required=True)
    parser.add_argument(
        "--identity-threshold",
        type=float,
        help=f"Cluster at this pairwise identity. Default: {COLD_PROTEIN_THRESHOLD} for "
        f"cold_protein; for cold_family the lowest of {FAMILY_THRESHOLD_CANDIDATES} "
        f"whose largest cluster holds at most {MAX_CLUSTER_FRACTION:.0%} of proteins.",
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    if args.identity_threshold is not None and not 0 < args.identity_threshold <= 1:
        parser.error("--identity-threshold must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    paths = paths_for(DATASET, SEQUENCE_PDB_SET, args.split)
    sequence_of = pdb_sequences(paths)
    sequences = sorted(set(sequence_of.values()))
    print(f"{len(sequence_of)} proteins, {len(sequences)} unique sequences")
    identity = cached_identity_matrix(paths.splits, sequences)

    largest_at = print_threshold_table(sequence_of, sequences, identity)
    threshold = args.identity_threshold
    if threshold is None:
        threshold = (
            COLD_PROTEIN_THRESHOLD
            if args.split == "cold_protein"
            else choose_family_threshold(largest_at, len(sequence_of))
        )
    print(f"{args.split}: identity threshold {threshold}")

    cluster_of = protein_clusters(sequence_of, sequences, identity, threshold)
    protein_rows = {
        name: rows.tolist()
        for name, rows in pd.read_csv(paths.csv).groupby("protein").indices.items()
    }
    train_folds, test_fold, test_clusters, fold_clusters = build_split(
        cluster_of, protein_rows, args.split_seed
    )

    held_out = sorted(name for name, c in cluster_of.items() if c in test_clusters)
    held_out_fraction = len(held_out) / len(cluster_of)
    if abs(held_out_fraction - TEST_FRACTION) > TEST_FRACTION_TOLERANCE:
        raise SystemExit(
            f"Held out {len(held_out)} proteins ({held_out_fraction:.1%}), more than "
            f"{TEST_FRACTION_TOLERANCE:.0%} from {TEST_FRACTION:.0%}"
        )
    info = {
        "split": args.split,
        "identity_threshold": threshold,
        "split_seed": args.split_seed,
        "sequence_pdb_set": SEQUENCE_PDB_SET,
        "n_proteins": len(cluster_of),
        "n_clusters": len(set(cluster_of.values())),
        "n_held_out_proteins": len(held_out),
        "held_out_fraction": held_out_fraction,
        "n_held_out_clusters": len(test_clusters),
        "held_out_proteins": held_out,
        "test_rows": len(test_fold),
        "fold_rows": [len(fold) for fold in train_folds],
        "fold_clusters": [len(clusters) for clusters in fold_clusters],
        "cluster_of": cluster_of,
    }
    write_split(paths, train_folds, test_fold, info)
    print(
        f"Held out {len(held_out)} proteins ({held_out_fraction:.1%}) in "
        f"{len(test_clusters)} clusters, {len(test_fold)} test rows; "
        f"fold rows {info['fold_rows']}"
    )
    print(f"Wrote {os.path.dirname(paths.train_folds)}")


if __name__ == "__main__":
    main()
