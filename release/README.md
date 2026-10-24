# Released results and cohort lists

This directory holds the data that pins the repository to the manuscript. All paths
and hashes are recorded in `src/pgl/release.json`, and `tests/test_release.py` checks
them, so the files cannot drift from the code without a failing test.

- `splits/` — sample identifiers of every evaluation split, one `sample_id` per row,
  sorted. A split CSV hashes to the value recorded in `release.json`. Structures and
  labels are not included; the identifiers are enough to check that a locally prepared
  cohort matches the released one.
- `tables/` — the released result tables: CASF-2016 for all retrained methods, and
  BDB2020+, LP-PDBBind and PDBbind-2013 per benchmark. These are the tables quoted in
  the manuscript and summarised in the README.
- `figures/fig2_casf2016_source.csv` — the source data of Fig. 2a/2b (per-method
  CASF-2016 RMSE and rank statistics). It is derived from
  `tables/exp1_main_casf2016.csv`; the test verifies row count, method order and the
  RMSE/Pearson values.
- `predictions/casf2016_pgl_seed20261022.csv` — predictions of the released readout
  (seed 20261022) on CASF-2016, with the test labels removed. The identifiers match
  the CASF-2016 split.
- `figure_data/` — per-panel source tables for Figs. 1–4 and Fig. S1, derived from the
  released tables and predictions; `figure_data/README.md` maps directories to panels.
  Panels that need coordinates (Fig. 1a/e, Fig. 3a) use the structure files named in the
  manuscript's provenance records; the structures themselves are not redistributed.

The three readout checkpoints for the released model (`readout.json`, one per seed) are
attached to the GitHub release `v1.0.0` as `PGL_weights_v1.0.0.zip`.

The tables were produced by the released evaluation pipeline with the three seeds
20261022, 20261023 and 20261024; metrics are seed means with sample standard
deviations. Reporting conventions and the definition of pK are in
`src/pgl/release.json` under `metrics`.
