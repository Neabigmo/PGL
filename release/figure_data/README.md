# Figure source data

These tables hold the per-panel values used for the manuscript figures. They are derived
from the released result tables and predictions in `release/tables/` and
`release/predictions/`, and from the feature extraction records of the released model.
They are included so that every panel can be redrawn without re-running extraction or
training. Panels that require molecular coordinates (Fig. 1a/e and Fig. 3a) additionally
use the structure files named in the accompanying provenance records; the coordinate
files themselves are not redistributed because the PDBbind and CASF-2016 structures are
distributed under their own licences.

| directory | figure | files |
|---|---|---|
| `fig1_operator/` | Fig. 1 | contact table, W and M-I matrices, quadratic forms, fixed-support response, local response, properties |
| `fig2_predictive/` | Fig. 2 | protocol matrix, calibration and residual bins, per-group paired errors, BDB transfer, LP/2013 |
| `fig3_structural/` | Fig. 3 | case selection and scores, per-target decisions, Top-k rates, paired differences, threshold curves |
| `fig4_computational/` | Fig. 4 | accuracy-cost, component composition, scaling |
| `si/` | Fig. S1 | coverage and matched metric control |

The figure-level `PROVENANCE.json` files in the manuscript package record the mapping
from every panel to its source CSV, the model/checkpoint identifier, units, sample size,
uncertainty convention and, for structural panels, the coordinate files and camera.
