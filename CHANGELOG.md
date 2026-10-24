# Changelog

## 1.0.0

First release matching the manuscript.

- `pgl` package with the persistent Gauss-linking operator and the
  287-dimensional contact-token representation.
- Frozen context interface for precomputed 512-column (Uni-Mol) and 480-column
  (SaProt) embeddings, assembled to 1,279 features.
- XGBoost readout with the predeclared hyper-parameters and three fixed seeds.
- Command line: `pgl features`, `pgl context`, `pgl train`, `pgl predict`,
  `pgl info`.
- `src/pgl/release.json` as the single release record: model identifier,
  feature schema, sample-list hashes, metric definitions and checkpoint layout.
- `release/schema.json` with the ordered 1,279 feature names and a schema hash;
  readouts can be bound to the schema and are checked against it on load.
- `release/splits/pdbbind2016_train3758.csv`, the context-complete PDBbind-2016
  training list, and `pgl train --train-manifest` to select it.
- Released data in `release/`: result tables, figure source data, split
  identifier lists, an example prediction file and an end-to-end regression
  fixture, all covered by tests.
- Documentation for the algorithm, the feature layout, the API and the four
  benchmark reproductions.
- Test suite and GitHub Actions workflow running on CPU.

## 1.0.0 fixes

- `scikit-learn` is a runtime dependency; the XGBoost readout imported it but did
  not declare it.
- Released CSVs are normalised to LF and `.gitattributes` keeps them LF, so the
  SHA-256 values recorded in `release.json` are platform independent.
- `pgl.metrics.spearman` returns NaN for constant input instead of emitting a
  SciPy warning, which keeps tiny demo fits warning-free.
- The earlier internal readout moved to `legacy/` and is no longer installed.
