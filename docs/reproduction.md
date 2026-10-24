# Benchmark reproduction

This note gives the commands that produce the four benchmark tables from
structures and labels. The same code path is used for every method family: contact
tokens, optional frozen context, one XGBoost readout, three fixed seeds.

## Environment

```bash
python -m pip install -e .
python -m pgl --version
```

Feature extraction and the readout are CPU-only. Context embeddings are produced
outside this repository (Uni-Mol 512 for the ligand, SaProt 480 for the protein);
the commands below assume their archives already exist.

## Expected data layout

```text
data/
  structures/<split>/<sample_id>_protein.pdb
  structures/<split>/<sample_id>_ligand.sdf
  manifests/<split>.csv
  labels/<split>.csv
  context/<split>_unimol.npz
  context/<split>_saprot.npz
```

`manifests/<split>.csv` has three columns `sample_id, protein_path, ligand_path`
with paths relative to the repository root or absolute. `labels/<split>.csv` has
`sample_id, label`, where `label` is pK = −log10(Kd/Ki). Test manifests are used
only after predictions are written; labels for a test split are joined in the
evaluation step and are never passed to `pgl train`.

`context/<split>_<tag>.npz` stores `ids` and `vectors`; `unimol` archives have 512
columns and `saprot` archives 480.

The four benchmark splits are:

| split name | rows | role |
|---|---|---|
| `dev` | 3,766 | PDBbind-2016 development: readout training |
| `casf2016_test` | 285 | CASF-2016 core set, external test |
| `bdb2020plus` | 115 | BDB2020+ external test |
| `lp_train`, `lp_val`, `lp_test` | 7,276 / 941 / 2,138 | LP-PDBBind official split |
| `pdbbind2013_train`, `pdbbind2013_val`, `pdbbind2013_test` | 2,095 / 540 / 193 | reconstructed PDBbind-2013 split |

## Released cohort lists and results

`release/splits/` contains the released identifiers of every split, with the SHA-256
of each file recorded in `src/pgl/release.json`. A locally prepared cohort matches the
released one when its `sample_id` column, sorted the same way, hashes to the same
value; `tests/test_release.py` performs this check for the bundled files.

`release/tables/` holds the released result tables and
`release/figures/fig2_casf2016_source.csv` the source data of Fig. 2. The example
predictions in `release/predictions/` are the CASF-2016 output of the released readout
(seed 20261022) without labels and use the identifiers of `release/splits/casf2016_test.csv`.

## Step 1: contact tokens

Run once per split. The command prints the number of featurized complexes and any
sample that was skipped because it has no contact in the central 5–6 Å window.

```bash
for split in dev casf2016_test bdb2020plus lp_train lp_val lp_test \
             pdbbind2013_train pdbbind2013_val pdbbind2013_test; do
  pgl features --manifest data/manifests/${split}.csv --out work/${split}_tokens.npz
done
```

## Step 2: frozen context

```bash
for split in dev casf2016_test bdb2020plus lp_train lp_val lp_test \
             pdbbind2013_train pdbbind2013_val pdbbind2013_test; do
  pgl context --tokens work/${split}_tokens.npz \
      --unimol data/context/${split}_unimol.npz \
      --saprot data/context/${split}_saprot.npz \
      --out work/${split}_features.npz
done
```

The result has 1,279 columns: 287 contact tokens, 512 Uni-Mol, 480 SaProt.

## Step 3: readout for CASF-2016 and BDB2020+

Both benchmarks apply the model trained on the development split, without
retraining. One model directory per seed:

```bash
for seed in 20261022 20261023 20261024; do
  pgl train --features work/dev_features.npz --labels data/labels/dev.csv \
      --out weights/pgl/seed${seed} --seed ${seed} --threads 8
done
```

```bash
for seed in 20261022 20261023 20261024; do
  pgl predict --model weights/pgl/seed${seed} \
      --features work/casf2016_test_features.npz \
      --out work/casf2016_seed${seed}.csv
  pgl predict --model weights/pgl/seed${seed} \
      --features work/bdb2020plus_features.npz \
      --out work/bdb2020plus_seed${seed}.csv
done
```

BDB2020+ labels are pKa on a different scale from the CASF pK values; report the
benchmark separately and never pool the two.

## Step 4: readout for LP-PDBBind

LP-PDBBind uses its own official training split with the same hyper-parameters;
the validation split is diagnostic and is not used for model selection.

```bash
for seed in 20261022 20261023 20261024; do
  pgl train --features work/lp_train_features.npz --labels data/labels/lp_train.csv \
      --out weights/lp/seed${seed} --seed ${seed} --threads 8
  pgl predict --model weights/lp/seed${seed} \
      --features work/lp_test_features.npz --out work/lp_seed${seed}.csv
done
```

The LP test split overlaps the PDBbind-2016 development set, so LP is a
covered-subset benchmark rather than an independent external set. Report it in its
own table and do not merge it with CASF-2016 or BDB2020+.

## Step 5: readout for PDBbind-2013

Same recipe with the reconstructed 2013 split:

```bash
for seed in 20261022 20261023 20261024; do
  pgl train --features work/pdbbind2013_train_features.npz \
      --labels data/labels/pdbbind2013_train.csv \
      --out weights/pdbbind2013/seed${seed} --seed ${seed} --threads 8
  pgl predict --model weights/pdbbind2013/seed${seed} \
      --features work/pdbbind2013_test_features.npz \
      --out work/pdbbind2013_seed${seed}.csv
done
```

PDBbind-2013 is a reconstructed list (release year ≤ 2012). Its test split shares
protein groups with training under the standard non-group split of that benchmark,
so the numbers are reported as a separate table. Predictions are written for every
test row; metrics use the rows whose geometry and context blocks are available.

## Step 6: metrics

The reported value is the mean over the three seeds. The snippet below computes the
per-seed and seed-averaged metrics from the prediction CSVs and the label CSV:

```python
import csv
import numpy as np
from pgl.metrics import all_metrics

def read_predictions(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: float(row["y_pred"]) for row in csv.DictReader(handle)}

def read_labels(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: float(row["label"]) for row in csv.DictReader(handle)}

labels = read_labels("data/labels/casf2016_test.csv")
seeds = (20261022, 20261023, 20261024)
predictions = [read_predictions(f"work/casf2016_seed{seed}.csv") for seed in seeds]
ids = sorted(set.intersection(*(set(pred) for pred in predictions)) & set(labels))

y = np.asarray([labels[sample_id] for sample_id in ids])
seeded = [np.asarray([pred[sample_id] for sample_id in ids]) for pred in predictions]
for seed, values in zip(seeds, seeded):
    print(seed, all_metrics(y, values))
print("mean prediction", all_metrics(y, np.mean(seeded, axis=0)))
```

The paper also reports paired comparisons between methods. Those resample protein
groups, so the manifest must carry a `protein_group` column; the comparison
statistic is the paired difference in mean squared error over 2,000 cluster
bootstrap draws with a fixed seed.

## Determinism

Feature extraction is deterministic and identical across repeated runs. XGBoost
training is deterministic for a fixed seed and thread count; the paper uses
`--threads 8`. Prediction from a saved readout is deterministic.

## Released figure and table sources

The released files are tied to the manuscript by the mapping below. All paths are
recorded in `src/pgl/release.json` with a SHA-256 hash.

| released file | source | manuscript use |
|---|---|---|
| `release/tables/exp1_main_casf2016.csv` | CASF-2016 evaluation of all retrained methods, 3 seeds | main comparison table, Fig. 2a/2b |
| `release/tables/exp2_bdb2020plus.csv` | BDB2020+ evaluation of the PDBbind-2016 readout | Fig. 2c, SI table |
| `release/tables/exp2_lp_pdbbind.csv` | LP-PDBBind retraining and evaluation | Fig. 2c, SI table |
| `release/tables/exp2_pdbbind2013.csv` | PDBbind-2013 retraining and evaluation | Fig. 2c, SI table |
| `release/figures/fig2_casf2016_source.csv` | columns `method, n_test, rmse_mean, rmse_std, pearson_mean, spearman_mean` of the `status == ok` rows of `exp1_main_casf2016.csv`, in table order | Fig. 2a/2b source data |
| `release/predictions/casf2016_pgl_seed20261022.csv` | output of the released readout on CASF-2016, seed 20261022, test labels removed | scatter panels and spot checks |
| `release/splits/*.csv` | `sample_id` column of each evaluation manifest, sorted | cohort verification |

Generation rules follow from the mapping. `fig2_casf2016_source.csv` is a column
subset of the table, not an independent computation: same methods, same order, same
number of rows (17 methods with `status == ok`), identical RMSE, RMSE standard
deviation, Pearson and Spearman values. The PGL row of each table is copied into
`release.json` under `results.benchmarks`, and the README benchmark table reports
those values rounded to three decimals. `tests/test_release.py` enforces all three
links.

Statistics that need per-complex protein groups (the 57-group ranking and the paired
cluster bootstrap) are not redistributed here; they are recomputed from the released
predictions and a local manifest that carries `protein_group`. Fig. 2d–2f therefore
use the same tables and predictions listed above with those local labels.

Any change to a result table therefore requires regenerating the figure source, the
README table and the hashes in `release.json` in the same commit; the release tests
fail otherwise.

