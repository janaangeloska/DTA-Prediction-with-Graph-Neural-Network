# DTA-Prediction-with-Graph-Neural-Network
This repository contains code and resources for replicatin the work described in the paper "Drug–Target Affinity Prediction using Graph Neural Network and Contact Maps" (RSC Advances, 2020). In this project, I changed the protein representation by using contact maps derived from AlphaFold-generated PDB files.

## Paths

Every input and output location derives from a single root, resolved in `Paths.py`. Leave
`DTA_DATA_ROOT` unset and everything resolves inside this checkout. Set it and the whole run moves,
which is what lets a Colab session pick up where the last one stopped instead of re-downloading and
regenerating from scratch.

On Colab, point it at a mounted Drive folder before importing any project module:

```python
from google.colab import drive

drive.mount("/content/drive")

import os

os.environ["DTA_DATA_ROOT"] = "/content/drive/MyDrive/dta"
```

The layout under the root is identical either way:

| Path | Holds |
|---|---|
| `data/davis/davis.csv` | affinity table |
| `data/davis/train_folds.txt`, `test_fold.txt` | community fold indices |
| `data/davis/pdb/` | AlphaFold structures, one per Davis name |
| `data/davis/contact_maps/` | contact maps, `.npz` |
| `data/davis/ligand_graphs/` | ligand graphs, `.gml` |
| `data/davis/protein_graphs/gml/` | protein graphs, `.gml` |
| `models_davis/` | best model and resume checkpoint, per fold |
| `results_davis/` | test metrics and density plot |

`training_5_folds.py` writes a resume checkpoint holding model state, optimizer state, epoch and
fold index, and restarts a fold from that checkpoint when one is present. With `DTA_DATA_ROOT` on
Drive those checkpoints outlive the session; without it they die with the VM.

## Data

`data/davis/` holds the committed Davis inputs:

- `davis.csv` holds 30,056 rows, columns `ligand` (SMILES), `protein` (Davis kinase name), `label` (pKd).
  Built from the GraphDTA release of Davis: `Y` holds Kd in nM, and the label is `-log10(Kd / 1e9)`,
  the same transform DeepDTA and GraphDTA use. Row order is row-major over the 68x442 affinity
  matrix, so row `i` is ligand `i // 442` and protein `i % 442`, which is the ordering the
  published fold indices assume.
- `train_folds.txt` and `test_fold.txt` are verbatim copies of GraphDTA's `train_fold_setting1.txt`
  and `test_fold_setting1.txt`. These are the standard community splits and are not reshuffled.

Everything else under `data/davis/` is generated and gitignored.

### Regenerating the protein graphs

1. Resolve each Davis protein name to a UniProt accession (strip any parenthetical, then query
   UniProt for the human reviewed entry). Nine names are aliases or non-human and need an explicit
   mapping: `ABL1p`→ABL1, `CDK4-cyclinD1`/`CDK4-cyclinD3`→CDK4, `PKAC-alpha`→PRKACA (P17612),
   `PKAC-beta`→PRKACB, `PFTAIRE2`→CDK15, `PFCDPK1`/`PFPK5`→*P. falciparum*, `PKNB`→*M. tuberculosis*.
2. Download `https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb` for each of the 375
   unique accessions. Note the version: the older `v4` URLs now return 404.
3. Save one PDB per Davis name as `data/davis/pdb/{name}.pdb`. The 75 variant names (for example
   `ABL1(E255K)`) reuse the wild-type structure, since AlphaFold only models the canonical sequence.
4. Run `python Protein_Representation.py`. It writes contact maps at an 8 A CA-CA cutoff to
   `data/davis/contact_maps/`, then residue graphs with 33 features per node to
   `data/davis/protein_graphs/gml/`.

### Regenerating the ligand graphs

Run `python Drug_Representation.py`. It reads `data/davis/davis.csv` and writes 68 molecular graphs
with 78 features per node to `data/davis/ligand_graphs/`, named after the sanitized SMILES.

