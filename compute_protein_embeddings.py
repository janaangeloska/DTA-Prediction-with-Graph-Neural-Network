import argparse
import os

import pandas as pd
import torch

from src.common.paths import DatasetPaths, paths_for
from src.common.structure_sets import add_pdb_set_arg
from src.sequence.protein_embeddings import (
    DEFAULT_SEQUENCE_SOURCE,
    MODEL_NAMES,
    build_adapter,
    cache_embedding,
    is_cached,
    load_pooled_embeddings,
    pdb_sizes,
    windowed_path,
    write_json,
    write_sequences,
)
from src.structure.protein_representation import pdb_sequence

DATASET = "davis"


def pdb_sequences(paths: DatasetPaths) -> dict[str, str]:
    names = pd.read_csv(paths.csv)["protein"].unique()
    return {name: pdb_sequence(paths.pdb_file(name)) for name in names}


def full_sequences(paths: DatasetPaths) -> dict[str, str]:
    raise SystemExit(
        f"Full-length sequences are not available: {paths.csv} holds kinase names "
        f"only and {paths.dataset_dir} has no name-to-sequence file (such as "
        "GraphDTA's proteins.txt). Use --sequence-source pdb."
    )


SEQUENCE_SOURCES = {"pdb": pdb_sequences, "full": full_sequences}


def print_pooled_stats(model_name: str, pooled: torch.Tensor) -> None:
    per_dim_std = pooled.std(dim=0)
    print(
        f"{model_name} pooled vectors over {pooled.shape[0]} unique sequences: "
        f"mean {pooled.mean():.4f}, std {pooled.std():.4f}, "
        f"max |x| {pooled.abs().max():.4f}, per-dimension std "
        f"min {per_dim_std.min():.4f} max {per_dim_std.max():.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Cache pooled protein language model embeddings for {DATASET}."
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--sequence-source",
        choices=sorted(SEQUENCE_SOURCES),
        default=DEFAULT_SEQUENCE_SOURCE,
        help="pdb: residues of the AlphaFold structures the GNN branch uses. "
        "full: full-length sequences.",
    )
    parser.add_argument(
        "--keep-residues",
        action="store_true",
        help="Also store the per-residue [L, D] tensor next to the pooled [D] vector.",
    )
    add_pdb_set_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = paths_for(DATASET, args.pdb_set)
    print(f"Structures: {args.pdb_set} ({paths.pdb})")
    sequences = SEQUENCE_SOURCES[args.sequence_source](paths)
    adapter = build_adapter(args.model)

    cache_dir = paths.protein_embeddings
    source_files = (
        pdb_sizes(paths.pdb_file, sequences) if args.sequence_source == "pdb" else {}
    )
    write_sequences(cache_dir, args.sequence_source, sequences, source_files)
    windowed = {
        name: len(seq) for name, seq in sequences.items() if adapter.needs_windows(seq)
    }
    write_json(
        {"max_residues": adapter.max_residues, "lengths": windowed},
        windowed_path(cache_dir, args.model, args.sequence_source),
    )
    if windowed:
        listing = ", ".join(f"{name} ({n})" for name, n in sorted(windowed.items()))
        print(
            f"{len(windowed)} proteins exceed {adapter.max_residues} residues and are "
            f"embedded in overlapping windows: {listing}"
        )

    unique = sorted(set(sequences.values()))
    missing = [
        seq
        for seq in unique
        if not is_cached(cache_dir, args.model, seq, args.keep_residues)
    ]
    print(
        f"{len(sequences)} proteins, {len(unique)} unique sequences, "
        f"{len(missing)} to embed with {args.model}"
    )
    if missing:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {args.model} on {device}")
        adapter.load(device)
        for i, seq in enumerate(missing, start=1):
            cache_embedding(adapter, cache_dir, args.model, seq, args.keep_residues)
            print(f"[{i}/{len(missing)}] embedded {len(seq)} residues")
        print(f"Embeddings in {os.path.join(cache_dir, args.model)}")

    embeddings = load_pooled_embeddings(
        cache_dir, args.model, paths.pdb_file, args.sequence_source
    )
    by_sequence = {sequences[name]: vector for name, vector in embeddings.items()}
    pooled = torch.stack([by_sequence[seq] for seq in unique])
    print_pooled_stats(args.model, pooled)


if __name__ == "__main__":
    main()
