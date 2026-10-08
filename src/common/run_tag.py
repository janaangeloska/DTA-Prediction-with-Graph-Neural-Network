import argparse

from src.common.seeding import SEED
from src.common.structure_sets import (
    DEFAULT_STRUCTURE_SET,
    STRUCTURE_SETS,
    add_pdb_set_arg,
)
from src.sequence.plm_net import PLMNet
from src.sequence.protein_embeddings import MODEL_NAMES
from src.structure.gnn_net import GNNNet

MODEL_CLASS_NAMES = {"structure": GNNNet.__name__, "plm": PLMNet.__name__}


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protein-repr",
        choices=sorted(MODEL_CLASS_NAMES),
        default="structure",
        help="structure: AlphaFold contact-map graph. "
        "plm: pooled embedding from a frozen protein language model.",
    )
    parser.add_argument(
        "--plm-model",
        choices=MODEL_NAMES,
        help="Protein language model, required with --protein-repr plm.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize each embedding dimension across proteins, plm only.",
    )
    add_pdb_set_arg(parser)
    parser.add_argument("--seed", type=int, default=SEED)


def check_run_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.protein_repr == "plm" and args.plm_model is None:
        parser.error(
            f"--plm-model is required with --protein-repr plm, "
            f"choose from: {', '.join(MODEL_NAMES)}"
        )
    if args.protein_repr != "plm" and args.plm_model is not None:
        parser.error("--plm-model only applies to --protein-repr plm")
    if args.protein_repr != "plm" and args.standardize:
        parser.error("--standardize only applies to --protein-repr plm")


def run_tag(
    protein_repr: str,
    plm_model: str | None,
    seed: int,
    standardize: bool = False,
    pdb_set: str = DEFAULT_STRUCTURE_SET,
) -> str:
    parts = [plm_model] if protein_repr == "plm" else []
    if STRUCTURE_SETS[pdb_set].tag:
        parts.append(STRUCTURE_SETS[pdb_set].tag)
    if standardize:
        parts.append("z")
    if seed != SEED:
        parts.append(f"s{seed}")
    return "_".join(parts)


def tag_suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""
