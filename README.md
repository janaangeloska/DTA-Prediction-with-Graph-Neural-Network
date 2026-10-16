# DTA-Prediction-with-Graph-Neural-Network
This repository contains code and resources for replicating the work described in the paper "Drug–Target Affinity Prediction using Graph Neural Network and Contact Maps" (RSC Advances, 2020). In this project, I changed the protein representation by using contact maps derived from AlphaFold-generated PDB files.

## Layout

- `src/common/`: paths, seeding, run tags, dataset building, checkpoints, metrics, prediction
  export, drug graphs (`drug_representation.py`) and the cold splits (`splits.py`).
- `src/structure/`: protein contact-map graphs (`protein_representation.py`) and the GNN model
  (`GNNNet` in `gnn_net.py`).
- `src/sequence/`: protein language model embeddings (`protein_embeddings.py`) and the embedding
  model (`PLMNet` in `plm_net.py`).
- `src/hybrid/`: the contact-map graph model with per-residue embeddings (`HybridNet` in
  `hybrid_net.py`).
- Entry scripts stay at the root: `training_5_folds.py`, `testing.py`, `report_best_epochs.py`,
  `compute_protein_embeddings.py`.

Run every command from the repository root. On Colab, `cd` into the repository folder first so
`src` is importable.

## Data

`data/` is not in git. Put these files in `data/davis/`:

- `davis.csv`: the 30,056 ligand-protein pairs of Davis, columns `ligand`, `protein`, `label`, built
  from the GraphDTA release with label `-log10(Kd / 1e9)`.
- `train_folds.txt` and `test_fold.txt`: GraphDTA's `train_fold_setting1.txt` and
  `test_fold_setting1.txt`, renamed.
- `pdb_kinase_domain/`: the 442 `.pdb` files from `prot_3d_for_Davis.tar.gz` in
  receptor-ai/3d-prot-dta, with their original names.

To keep data and outputs somewhere else, for example a mounted Google Drive folder on Colab, set
`DTA_DATA_ROOT` to that folder before running any script.

Without `--pdb-set`, the scripts use the full-length structures in `data/davis/pdb/`, kept only to
reproduce earlier runs.

## Build the graphs

```bash
python -m src.common.drug_representation
python -m src.structure.protein_representation --pdb-set kinase_domain
```

## Train and test

```bash
python training_5_folds.py --pdb-set kinase_domain
python report_best_epochs.py --pdb-set kinase_domain
python training_5_folds.py --pdb-set kinase_domain --mode full --epochs N
python testing.py --pdb-set kinase_domain
python testing.py --pdb-set kinase_domain --full
```

`N` is the median best epoch printed by `report_best_epochs.py`. Models and checkpoints go to
`models_davis/`, metrics and plots to `results_davis/`. A fold restarts from its checkpoint if one
exists.

`testing.py --save-predictions` also writes the per-row test predictions to `results_davis/`, and
`--predictions-only` writes only those. `--seed` sets the run seed, and a non-default seed is added
to the output file names.

## Splits

`--split` picks the folds: `random` (default, the train and test folds above), `cold_protein`
(whole clusters of near-identical protein sequences held out) or `cold_family` (whole clusters of
related kinases held out). The cold splits are built once from the `kinase_domain` sequences and
written to `data/davis/splits/<split>/`:

```bash
python -m src.common.splits --split cold_protein
python -m src.common.splits --split cold_family
python training_5_folds.py --pdb-set kinase_domain --split cold_protein
```

Reporting, `--mode full` and testing take the same `--split` flag.

## Protein language model arm

Replaces the protein graph with the mean-pooled embedding of a frozen protein language model:
`esmc_300m`, `esmc_600m`, `prot_t5_xl_uniref50`, `esm2_650m` or `protbert`. ESM C needs
`pip install esm`, the others `pip install transformers sentencepiece`. `random_1024` is a control:
a fixed random vector per sequence, so it carries protein identity and nothing else. `--standardize`
standardizes each embedding dimension across proteins.

```bash
python compute_protein_embeddings.py --model esmc_300m --pdb-set kinase_domain
python training_5_folds.py --protein-repr plm --plm-model esmc_300m --pdb-set kinase_domain
```

Reporting, `--mode full` and testing take the same flags as above.

## Hybrid arm

Keeps the contact-map graph and GNNNet's protein trunk, and appends a learned 64-d projection of
each residue's frozen language model embedding to the 33 base node features (`HybridNet` in
`src/hybrid/hybrid_net.py`). `hybrid_noedges` keeps only self-loops and `hybrid_shuffled` rewires
the contacts at random with every node degree kept, as controls for the edges.

```bash
python compute_protein_embeddings.py --model esmc_300m --pdb-set kinase_domain --keep-residues
python training_5_folds.py --protein-repr hybrid --plm-model esmc_300m --pdb-set kinase_domain
```

Swap `hybrid` for `hybrid_noedges` or `hybrid_shuffled` for the controls. Reporting, `--mode full`
and testing take the same flags. `random_1024` has no per-residue embeddings, so it does not work
with the hybrid arms.
