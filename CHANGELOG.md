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
- Released data in `release/`: result tables, Fig. 2 source data, split
  identifier lists and an example prediction file, all covered by tests.
- Documentation for the algorithm, the feature layout, the API and the four
  benchmark reproductions.
- Test suite and GitHub Actions workflow running on CPU.
- The earlier internal readout moved to `legacy/` and no longer installed.
