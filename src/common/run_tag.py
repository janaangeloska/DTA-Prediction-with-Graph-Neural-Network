import argparse

from src.common.seeding import SEED
from src.common.split_sets import DEFAULT_SPLIT, SPLITS, add_split_arg
from src.common.structure_sets import (
    DEFAULT_STRUCTURE_SET,
    STRUCTURE_SETS,
    add_pdb_set_arg,
)
from src.hybrid.hybrid_net import HybridNet
from src.sequence.plm_net import PLMNet
from src.sequence.protein_embeddings import MODEL_NAMES
from src.structure.gnn_net import GNNNet

HYBRID_REPRS = ("hybrid", "hybrid_noedges", "hybrid_shuffled")
EMBEDDING_REPRS = ("plm", *HYBRID_REPRS)

MODEL_CLASS_NAMES = {
    "structure": GNNNet.__name__,
    "plm": PLMNet.__name__,
    **dict.fromkeys(HYBRID_REPRS, HybridNet.__name__),
}


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protein-repr",
        choices=sorted(MODEL_CLASS_NAMES),
        default="structure",
        help="structure: AlphaFold contact-map graph. "
        "plm: pooled embedding from a frozen protein language model. "
        "hybrid: contact-map graph with per-residue embeddings added to the nodes. "
        "hybrid_noedges: the same nodes with self-loops only. "
        "hybrid_shuffled: the same nodes on a degree-preserving rewiring of the contacts.",
    )
    parser.add_argument(
        "--plm-model",
        choices=MODEL_NAMES,
        help=f"Protein language model, required with --protein-repr "
        f"{', '.join(EMBEDDING_REPRS)}.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize each embedding dimension across proteins, plm only.",
    )
    add_pdb_set_arg(parser)
    add_split_arg(parser)
    parser.add_argument("--seed", type=int, default=SEED)


def check_run_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    uses_embeddings = args.protein_repr in EMBEDDING_REPRS
    if uses_embeddings and args.plm_model is None:
        parser.error(
            f"--plm-model is required with --protein-repr {args.protein_repr}, "
            f"choose from: {', '.join(MODEL_NAMES)}"
        )
    if not uses_embeddings and args.plm_model is not None:
        parser.error(
            f"--plm-model only applies to --protein-repr {', '.join(EMBEDDING_REPRS)}"
        )
    if args.protein_repr != "plm" and args.standardize:
        parser.error("--standardize only applies to --protein-repr plm")


def run_tag(
    protein_repr: str,
    plm_model: str | None,
    seed: int,
    standardize: bool = False,
    pdb_set: str = DEFAULT_STRUCTURE_SET,
    split: str = DEFAULT_SPLIT,
) -> str:
    parts = [protein_repr] if protein_repr in HYBRID_REPRS else []
    if protein_repr in EMBEDDING_REPRS:
        parts.append(plm_model)
    if STRUCTURE_SETS[pdb_set].tag:
        parts.append(STRUCTURE_SETS[pdb_set].tag)
    if SPLITS[split].tag:
        parts.append(SPLITS[split].tag)
    if standardize:
        parts.append("z")
    if seed != SEED:
        parts.append(f"s{seed}")
    return "_".join(parts)


def tag_suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""
