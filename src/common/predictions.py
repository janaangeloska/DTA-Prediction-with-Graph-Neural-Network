from collections.abc import Sequence

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = [
    "row_index",
    "protein",
    "ligand",
    "label",
    "prediction",
    "source",
    "tag",
    "seed",
    "split",
]


def prediction_table(
    csv_file: str,
    test_indices: Sequence[int],
    labels: np.ndarray,
    predictions: np.ndarray,
    source: int | str,
    tag: str,
    seed: int,
    split: str,
) -> pd.DataFrame:
    """One row per scored test pair, keyed by its row in the dataset csv."""
    pairs = pd.read_csv(csv_file).iloc[list(test_indices)]
    expected = pairs["label"].to_numpy(dtype=np.float32)
    if len(expected) != len(labels) or not np.array_equal(expected, labels):
        raise ValueError(
            f"Scored labels do not match {csv_file} at the test indices, "
            "the predictions cannot be aligned to dataset rows."
        )
    table = pd.DataFrame(
        {
            "row_index": pairs.index.to_numpy(),
            "protein": pairs["protein"].to_numpy(),
            "ligand": pairs["ligand"].to_numpy(),
            "label": labels.astype(np.float64),
            "prediction": predictions.astype(np.float64),
            "source": source,
            "tag": tag,
            "seed": seed,
            "split": split,
        }
    )
    return table[PREDICTION_COLUMNS]
