import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitSet:
    """One way of dividing the pairs into training folds and a test fold."""

    description: str
    dir_name: str
    tag: str


SPLITS = {
    "random": SplitSet(
        description="random pairs, the DeepDTA/GraphDTA/DGraphDTA folds",
        dir_name="",
        tag="",
    ),
    "cold_protein": SplitSet(
        description="whole clusters of near-identical protein sequences held out",
        dir_name="cold_protein",
        tag="cprot",
    ),
    "cold_family": SplitSet(
        description="whole clusters of related kinases held out",
        dir_name="cold_family",
        tag="cfam",
    ),
}

DEFAULT_SPLIT = "random"

COLD_SPLITS = tuple(name for name, split in SPLITS.items() if split.dir_name)


def add_split_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        choices=list(SPLITS),
        default=DEFAULT_SPLIT,
        help="; ".join(
            f"{name}: {split.description}" for name, split in SPLITS.items()
        ),
    )
