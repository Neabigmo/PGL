# Released results, cohorts and fixtures

This directory holds the data that pins the repository to the manuscript. All paths
and hashes are recorded in `src/pgl/release.json`, and `tests/test_release.py` checks
them, so the files cannot drift from the code without a failing test.

- `splits/` — sample identifiers of every evaluation split, one `sample_id` per row.
  `pdbbind2016_train3758.csv` is the context-complete training subset (3,758 of the
  3,766 development rows) and keeps the order of the released parity record; the other
  lists are sorted. Each entry in `release.json` records two hashes: `sha256` over the
  id payload (one id per line, file order, no trailing newline) and `sha256_file` over
  the LF-normalised CSV bytes. Structures and labels are not included; the identifiers
  are enough to check that a locally prepared cohort matches the released one.
- `tables/` — the released result tables: CASF-2016 for all retrained methods, and
  BDB2020+, LP-PDBBind and PDBbind-2013 per benchmark. These are the tables quoted in
  the manuscript and summarised in the README.
- `figures/fig2_casf2016_source.csv` — pre-V5 aggregate source data of the CASF-2016
  comparison (per-method RMSE and rank statistics), derived from
  `tables/exp1_main_casf2016.csv`. The current Fig. 2 (V5 panel contract) uses the
  per-panel files in `figure_data/fig2_predictive/`.
- `predictions/casf2016_pgl_seed20261022.csv` — predictions of the released readout
  (seed 20261022) on CASF-2016, with the test labels removed. The identifiers match
  the CASF-2016 split.
- `figure_data/` — per-panel source tables for Figs. 1–4 and Fig. S1, derived from the
  released tables and predictions; `figure_data/README.md` maps directories to panels.
  Panels that need coordinates (Fig. 1a/e, Fig. 3a) use the structure files named in the
  manuscript's provenance records; the structures themselves are not redistributed.
- `schema.json` — the ordered 1,279 feature names (287 contact tokens, 512 Uni-Mol,
  480 SaProt) and their SHA-256. A readout saved with this schema records the hash and
  is checked against it on load.
- `fixtures/` — the end-to-end regression fixture: contact tokens of the bundled toy
  complex, a fixed synthetic context pair and a small public readout. It is used only
  by `tests/test_regression.py` to pin the code–feature–context–readout chain and is not
  a benchmark model.

The three readout checkpoints for the released model (`readout.json`, one per seed) are
attached to the GitHub release `v1.0.0` as `PGL_weights_v1.0.0.zip`.

The tables were produced by the released evaluation pipeline with the three seeds
20261022, 20261023 and 20261024; metrics are seed means with sample standard
deviations. Reporting conventions and the definition of pK are in
`src/pgl/release.json` under `metrics`. All released CSV files use LF line endings;
`.gitattributes` keeps them LF on every platform so the recorded hashes match.
