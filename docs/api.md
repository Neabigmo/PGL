# API reference

Everything below is importable from `pgl`.

## Contact tokens

```python
from pgl import contact_features, feature_names, FEATURE_DIM

result = contact_features("protein.pdb", "ligand.sdf")
result.vector        # (287,) float32
result.token_matrix  # (n_contacts, 57) float64
result.counts        # contact counts, edge counts, Betti numbers, chain errors
```

`contact_features(protein_path, ligand_path, complex_=None, windows=None)` accepts a
pre-parsed complex and pre-built windows to avoid parsing twice when several
representations of one structure are needed.

Lower-level entry points:

```python
from pgl import parse_complex, build_windows

complex_ = parse_complex("protein.pdb", "ligand.sdf")
windows = build_windows(complex_)          # three filtration windows
window = windows[1]                        # 5-6 A, used by the contact tokens
window.eigenvalues, window.eigenvectors    # operator spectrum
window.coupling                            # Gauss coupling of the contacts
```

`pgl.features.feature_names()` returns all 287 column names;
`pgl.features.BLOCK_SLICES` gives the three block offsets.

## Frozen context

```python
from pgl import ContextEmbedding, build_feature_matrix

unimol = ContextEmbedding.load("unimol.npz", "unimol")   # 512 columns
saprot = ContextEmbedding.load("saprot.npz", "saprot")   # 480 columns
matrix, names, blocks = build_feature_matrix(
    tokens, ids, {"unimol": unimol, "saprot": saprot}
)
```

`matrix` has 1,279 columns. Missing samples stay NaN. Pass `missing_flags=True`
to append one presence indicator per block.

`ContextEmbedding.from_mapping(tag, {sample_id: vector})` builds a block from
vectors already in memory.

## Readout

```python
from pgl import XGBReadout

readout = XGBReadout(seed=20261022, n_threads=8)
readout.fit(features, labels)          # labels in pK
predictions = readout.predict(features)

readout.save("weights/seed20261022")
loaded = XGBReadout.load("weights/seed20261022")
```

`XGBReadout.hyperparameters` holds the predeclared configuration
(`pgl.readout.XGB_PRESET`); pass `hyperparameters={...}` only for ablation runs.
`readout.config()` is a serialisable record of the fit.

## Metrics

```python
from pgl.metrics import all_metrics

all_metrics(y_true, y_pred)  # n, rmse, mae, pearson, spearman
```

## Archives and labels

```python
from pgl.data import load_tokens, save_tokens, load_features, load_labels

tokens = load_tokens("tokens.npz")        # (ids, tokens)
labels = load_labels("labels.csv")        # {sample_id: pK}
```

## Release record

```python
from pgl.release import RELEASE, MODEL_ID, RELEASE_VERSION, FEATURE_BLOCKS, release_file

RELEASE_VERSION, MODEL_ID        # "1.0.0", "PGL-1.0.0"
FEATURE_BLOCKS                   # [("contact_tokens", 287), ("ctx_unimol", 512), ("ctx_saprot", 480)]
RELEASE["metrics"]["label"]      # pK definition and metric conventions
path = release_file(RELEASE["results"]["predictions"]["casf2016_seed20261022"]["path"])
```

`release_file` resolves a path recorded in `src/pgl/release.json` against the source
checkout; it raises for an installed wheel, where the released data files are absent.

## Command line

```text
pgl features --protein P --ligand L --out tokens.npz [--id ID]
pgl features --manifest manifest.csv --out tokens.npz
pgl context  --tokens tokens.npz [--unimol U.npz] [--saprot S.npz] --out features.npz [--present-flags]
pgl train    --features F.npz --labels L.csv --out DIR [--seed 20261022] [--threads 8]
pgl predict  --model DIR --features F.npz --out predictions.csv
pgl info
```

`pgl features` accepts single-complex mode or a manifest with columns
`sample_id, protein_path, ligand_path`. Samples without a contact in the central
window are reported and skipped. `pgl train` and `pgl predict` accept either a
tokens archive or a features archive.
