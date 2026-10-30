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

## Notebooks

`notebooks/` holds the Colab notebooks that ran every experiment in the thesis, with their outputs
saved. They need Google Colab with a GPU. They do not download the code. Each one runs it from a
Google Drive folder that holds a copy of the four entry scripts and `src/`, and checks at the start
that the files are there. See [notebooks/README.md](notebooks/README.md) for the folders and the
order.

- `01_embeddings_esmc600m_prott5.ipynb`: computes the pooled ESM C 600M and ProtT5-XL embeddings.
- `02_random_split_arms.ipynb`: trains and tests the ProtT5-XL and structure arms on the random
  split, then compares all four arms.
- `03_random_split_seeds.ipynb`: retrains the full-data models with seeds 43 and 44.
- `04_controls.ipynb`: the drug-mean plus protein-mean baseline and the `random_1024` control.
- `05_cold_protein.ipynb`: all five arms on `cold_protein`.
- `06_cold_family.ipynb`: all five arms on `cold_family`.
- `07_hybrid.ipynb`: `hybrid` and `hybrid_noedges` on `cold_protein`.
- `08_error_analysis.ipynb`: re-scores the finished cold-split models with `--predictions-only`,
  then scores per protein, per cluster and with a paired cluster bootstrap.

Notebooks 01 to 03 ran on an earlier flat layout, with modules such as `Paths`, `RunTag` and
`Seeding` at the top level. Those files are not in the repository, so 01 to 03 do not run against
the current code as they are. Notebooks 04 to 08 import from `src/`.

The hybrid arm was run on `cold_protein` only. `hybrid_shuffled` is in the code but was not run.

Notebook 08 writes its tables to an `analysis/` folder in its Drive folder:
`per_protein_summary.csv`, `per_cluster_cold_protein.csv`, `per_cluster_cold_family.csv` and
`bootstrap_differences.csv`.

## Data

`data/` is not in git, apart from the cold splits in `data/davis/splits/`. Put these files in
`data/davis/`:

- `davis.csv`: the 30,056 ligand-protein pairs of Davis, columns `ligand`, `protein`, `label`, with
  label `-log10(Kd / 1e9)`. Built from the
  [DGraphDTA data folder](https://github.com/595693085/DGraphDTA/tree/master/data). Its README and
  the paper link to the original datasets.
- `train_folds.txt` and `test_fold.txt`: `train_fold_setting1.txt` and `test_fold_setting1.txt`
  from the same DGraphDTA folder, renamed.
- `pdb_kinase_domain/`: the 442 AlphaFold `.pdb` files from `prot_3d_for_Davis.tar.gz` in the
  [receptor-ai/3d-prot-dta data folder](https://github.com/receptor-ai/3d-prot-dta/tree/master/data),
  with their original names.

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
`models_davis/`, metrics and plots to `results_davis/`. A fold or full-data run resumes from its
checkpoint if one exists.

`testing.py --save-predictions` also writes the per-row test predictions to `results_davis/`, and
`--predictions-only` writes only those. `--seed` sets the run seed, and a non-default seed is added
to the output file names.

## Splits

`--split` picks the folds: `random` (default, the train and test folds above), `cold_protein`
(whole clusters of near-identical protein sequences held out) or `cold_family` (whole clusters of
related kinases held out). The cold splits are built once from the `kinase_domain` sequences and
written to `data/davis/splits/<split>/`. The built files are included in the repository. The same
command regenerates them. If the files already exist, it only checks that the new build matches
and stops if it does not.

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
