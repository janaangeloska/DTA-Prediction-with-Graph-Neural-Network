# Notebooks

These are the Google Colab notebooks that ran every experiment in the thesis. Each one has its results saved in it, so you can read them without running anything.

**The question:** to predict how strongly a drug binds a protein, is it better to give the model the protein's 3D structure, or just its amino acid sequence?

**The data:** Davis, 442 proteins and 68 drugs, which makes 30,056 drug-protein pairs.

**Two ways of giving the model a protein:**
- **Structure:** a graph built from the AlphaFold 3D structure of the protein.
- **Sequence:** a fixed vector from a pretrained protein language model (ESM C 300M, ESM C 600M or ProtT5-XL). The language model is not trained further.

## How to run

Run the notebooks in order, on Colab with a GPU. Each one uses a Google Drive folder. Notebooks 02 to 08 read files that earlier notebooks wrote, so the order matters.

| Notebook | Drive folder |
|---|---|
| 01, 02, 03 | `Diplomska_v2` |
| 04 | `Diplomska_ctrl` |
| 05, 06 | `Diplomska_cold` |
| 07, 08 | `Diplomska_hybrid` |

## What each notebook does

**01. Pooled embeddings**
Turns each protein sequence into one vector with ESM C 600M and ProtT5-XL.
Why: the models in the next notebooks need these vectors as input.

**02. Random split: ProtT5-XL and structure**
Trains and tests the ProtT5-XL model and the structure model, then compares all four arms on a random split of the pairs.
Why: this is the first fair comparison between structure and sequence.

**03. Random split: extra seeds**
Trains the full models again with two more random seeds (43 and 44).
Why: one run can be lucky. Three runs show how much a result changes by chance.

**04. Controls**
Two checks with no real protein information. The first predicts from drug and protein averages only. The second gives every protein a random vector.
Why: on the random split every protein is in both train and test, so a model can score well just by recognizing the protein. The controls show how much of the score comes from that.

**05. Cold protein split**
Holds out 88 whole proteins, so the test proteins are never seen in training. Near duplicates stay on the same side.
Why: this is the realistic case, predicting for a protein the model has never seen.

**06. Cold family split**
Holds out 88 proteins as whole protein families, so no test protein is similar to a training protein.
Why: this is the hardest test. It shows whether a model learned anything that carries over to new protein families.

**07. Hybrid arm**
Gives the structure graph the language model's per-residue features as extra input. Runs once with the graph edges and once without. Cold protein split only.
Why: to check if the 3D structure adds anything once the model already has the sequence information.

**08. Error analysis**
Re-scores the finished models on the cold splits and looks at the predictions protein by protein and family by family. It also compares arms with a bootstrap, which shows how sure we can be, since it accounts for which proteins happened to be held out.
Why: one overall score hides where a model fails. This shows where, and whether a difference between two arms is real.

## Short summary of what was found

- On the random split, all arms score about the same. The random vector control gets close to them.
- On unseen proteins, the sequence models (ESM C 300M and 600M) beat the drug-average baseline. The structure model does not.
- Adding the structure to the sequence features did not help.

## Things to know

- Some runs were resumed after Colab stopped. These are the seed 42 structure and ESM C 600M random-split models, the structure five-fold runs in 02, and ProtT5-XL on the cold family split.
- The hybrid arm was not run on the cold family split. Details are in notebook 07.
- Each notebook prints a check at the start. The lines marked "expected" should match what is printed.
