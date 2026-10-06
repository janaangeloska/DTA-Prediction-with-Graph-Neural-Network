# DTA-Prediction-with-Graph-Neural-Network
This repository contains code and resources for replicatin the work described in the paper "Drug–Target Affinity Prediction using Graph Neural Network and Contact Maps" (RSC Advances, 2020). In this project, I changed the protein representation by using contact maps derived from AlphaFold-generated PDB files.

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

## Build the graphs

```bash
python Drug_Representation.py
python Protein_Representation.py --pdb-set kinase_domain
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

## Protein language model arm

Replaces the protein graph with the mean-pooled embedding of a frozen protein language model:
`esmc_300m`, `esmc_600m`, `prot_t5_xl_uniref50`, `esm2_650m` or `protbert`. ESM C needs
`pip install esm`, the others `pip install transformers sentencepiece`.

```bash
python compute_protein_embeddings.py --model esmc_300m --pdb-set kinase_domain
python training_5_folds.py --protein-repr plm --plm-model esmc_300m --pdb-set kinase_domain
```

Reporting, `--mode full` and testing take the same flags as above.

Without `--pdb-set`, the scripts use the full-length structures in `data/davis/pdb/`, kept only to
reproduce earlier runs.
