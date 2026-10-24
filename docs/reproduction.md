# Benchmark reproduction

This note gives the commands that produce PGL's four benchmark tables from structures
and labels. The same code path is used for every PGL benchmark: contact tokens,
optional frozen context, one XGBoost readout, three fixed seeds. The comparison
methods in the manuscript are not reproduced by this repository; only the PGL path is
described here.

## Environment

```bash
python -m pip install -e .
python -m pgl --version
```

Contact-token extraction, context assembly and the readout are CPU-only. The two
context blocks are produced outside this repository by the Uni-Mol and SaProt scripts
(see the "Manuscript 1,279-D pipeline" section of the README); the commands below
assume their archives already exist.

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
`sample_id, label`, where `label` is pK = −log10(Kd/Ki). Test labels are read only in
the evaluation step, after predictions have been written; a test label is never
passed to `pgl train`.

`context/<split>_<tag>.npz` stores `ids` and `vectors`; `unimol` archives have 512
columns and `saprot` archives 480.

The PGL benchmark splits are:

| split name | rows | role |
|---|---|---|
| `pdbbind2016_dev` | 3,766 | PDBbind-2016 development manifest |
| `pdbbind2016_train3758` | 3,758 | development rows with both context blocks: readout training |
| `casf2016_test` | 285 | CASF-2016 core set, external test |
| `bdb2020plus` | 115 | BDB2020+ external test |
| `lp_pdbbind_train`, `lp_pdbbind_val`, `lp_pdbbind_test` | 7,276 / 941 / 2,138 | LP-PDBBind official split |
| `pdbbind2013_train`, `pdbbind2013_val`, `pdbbind2013_test` | 2,095 / 540 / 193 | reconstructed PDBbind-2013 split |

## Released cohort lists and results

`release/splits/` contains the released identifiers of every split. Two hashes are
recorded per split in `src/pgl/release.json`: `sha256` over the sample_id payload
(one id per line in file order, no trailing newline) and `sha256_file` over the
LF-normalised CSV file. The 3,758-row training list preserves the order of the
released parity record; its payload hash is
`6a42c6f4d238da1b53e4eb5de14a4ad081b6662fcaf209ea03abe5b735b2734a`.
`release/schema.json` lists the 1,279 feature names in column order with its own
SHA-256; `release/fixtures/` holds the end-to-end regression fixture.
`tests/test_release.py` verifies all of these files.

`release/tables/` holds the released result tables and `release/figure_data/` the
per-panel source data of the manuscript figures. The example predictions in
`release/predictions/` are the CASF-2016 output of the released readout (seed
20261022) without labels and use the identifiers of
`release/splits/casf2016_test.csv`.

## Step 1: contact tokens

Run once per split. The command prints the number of featurized complexes and any
sample that was skipped because it has no contact in the central 5–6 Å window.

```bash
for split in pdbbind2016_dev casf2016_test bdb2020plus lp_train lp_val lp_test \
             pdbbind2013_train pdbbind2013_val pdbbind2013_test; do
  pgl features --manifest data/manifests/${split}.csv --out work/${split}_tokens.npz
done
```

## Step 2: frozen context

```bash
for split in pdbbind2016_dev casf2016_test bdb2020plus lp_train lp_val lp_test \
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
retraining. The training cohort is the 3,758-row context-complete subset of the
3,766-row development manifest: training drops rows that lack either context block,
whereas inference keeps every test row and lets XGBoost route the NaN context columns.
The released list is passed with `--train-manifest` and checked against the schema:

```bash
for seed in 20261022 20261023 20261024; do
  pgl train --features work/pdbbind2016_dev_features.npz \
      --labels data/labels/pdbbind2016_dev.csv \
      --train-manifest release/splits/pdbbind2016_train3758.csv \
      --schema release/schema.json \
      --out weights/pgl/seed${seed} --seed ${seed} --threads 8
done
```

If the feature archive is already restricted to the training rows, `--train-manifest`
is not needed; `pgl train` then trains on every row of the archive.

```bash
for seed in 20261022 20261023 20261024; do
  pgl predict --model weights/pgl/seed${seed} --schema release/schema.json \
      --features work/casf2016_test_features.npz \
      --out work/casf2016_seed${seed}.csv
  pgl predict --model weights/pgl/seed${seed} --schema release/schema.json \
      --features work/bdb2020plus_features.npz \
      --out work/bdb2020plus_seed${seed}.csv
done
```

BDB2020+ labels are pKa values on the same negative-log scale as the other
benchmarks (pKa = 9 − log10 K_nM = −log10 K_M). The benchmark differs from CASF-2016
in assay source and cohort, not in the logarithmic unit; report it as a separate
benchmark and never pool it with the other datasets.

## Step 4: readout for LP-PDBBind

LP-PDBBind uses its own official training split with the same hyper-parameters;
the validation split is diagnostic and is not used for model selection.

```bash
for seed in 20261022 20261023 20261024; do
  pgl train --features work/lp_train_features.npz --labels data/labels/lp_train.csv \
      --schema release/schema.json \
      --out weights/lp/seed${seed} --seed ${seed} --threads 8
  pgl predict --model weights/lp/seed${seed} --schema release/schema.json \
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
      --schema release/schema.json \
      --out weights/pdbbind2013/seed${seed} --seed ${seed} --threads 8
  pgl predict --model weights/pdbbind2013/seed${seed} --schema release/schema.json \
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

The released files are tied to the manuscript by the mapping below. Result tables,
figure source files and the schema are recorded in `src/pgl/release.json` with their
SHA-256 hashes.

| released file | source | manuscript use |
|---|---|---|
| `release/tables/exp1_main_casf2016.csv` | CASF-2016 evaluation of all retrained methods, 3 seeds | main comparison table |
| `release/tables/exp2_bdb2020plus.csv` | BDB2020+ evaluation of the PDBbind-2016 readout | Fig. 2 (V5) dataset panel, SI table |
| `release/tables/exp2_lp_pdbbind.csv` | LP-PDBBind retraining and evaluation | Fig. 2 (V5) dataset panel, SI table |
| `release/tables/exp2_pdbbind2013.csv` | PDBbind-2013 retraining and evaluation | Fig. 2 (V5) dataset panel, SI table |
| `release/figure_data/fig2_predictive/` | per-panel CSVs for the current Fig. 2 (V5 panel contract); see its `PROVENANCE.json` | Fig. 2 (V5) |
| `release/figures/fig2_casf2016_source.csv` | pre-V5 aggregate: columns `method, n_test, rmse_mean, rmse_std, pearson_mean, spearman_mean` of the `status == ok` rows of `exp1_main_casf2016.csv` | legacy mapping kept for older figure drafts |
| `release/predictions/casf2016_pgl_seed20261022.csv` | output of the released readout on CASF-2016, seed 20261022, test labels removed | scatter panels and spot checks |
| `release/splits/*.csv` | `sample_id` column of each evaluation manifest (sorted; the 3,758 training list preserves parity order) | cohort verification |
| `release/schema.json` | ordered 1,279 feature names and their hash | checkpoint/schema verification |
| `release/fixtures/` | toy complex tokens, fixed context and a small public readout | end-to-end regression test |

The current figure set is being moved to the V5 panel contract; the per-panel CSVs in
`release/figure_data/fig2_predictive/` are the source of the V5 Fig. 2 panels, while
`release/figures/fig2_casf2016_source.csv` is the pre-V5 aggregate kept for the older
draft. When the V5 figure is frozen, this table will name the exact panels. Generation
rules for the aggregate file still hold: it is a column subset of the result table,
not an independent computation; same methods, same order, identical RMSE, RMSE
standard deviation, Pearson and Spearman values. The PGL row of each table is copied
into `release.json` under `results.benchmarks`, and the README benchmark table reports
those values rounded to three decimals. `tests/test_release.py` enforces these links.

Statistics that need per-complex protein groups (the 57-group ranking and the paired
cluster bootstrap) are not redistributed here; they are recomputed from the released
predictions and a local manifest that carries `protein_group`.

Any change to a result table requires regenerating the figure source, the README table
and the hashes in `release.json` in the same commit; the release tests fail otherwise.
