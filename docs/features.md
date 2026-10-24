# Feature layout

The representation of one complex has three blocks and 287 columns. With frozen
context embeddings attached the readout input grows to 1,279 columns.

| columns | block | width |
|---|---|---|
| 0–14 | chemistry channels | 15 |
| 15–128 | contact-token statistics | 114 |
| 129–286 | all-atom contacts | 158 |
| 287–798 | Uni-Mol ligand context (optional) | 512 |
| 799–1278 | SaProt protein context (optional) | 480 |

`pgl.features.feature_names()` returns the 287 names in this order;
`pgl.features.BLOCK_SLICES` holds the offsets. Values are stored as `float32`
after computation in float64. The same layout is recorded machine-readably in
`src/pgl/release.json` under `feature_schema`; the tests compare the two.

## 1. Chemistry channels (15, columns 0–14)

Each protein-ligand contact contributes a one-hot chemistry vector; the block
averages over contacts. The first five entries encode the residue class of the
protein atom:

| column | class | residues |
|---|---|---|
| 0 | hydrophobic | ALA, VAL, ILE, LEU, MET, PRO, GLY |
| 1 | polar | SER, THR, ASN, GLN, CYS |
| 2 | positive | LYS, ARG, HIS |
| 3 | negative | ASP, GLU |
| 4 | aromatic | PHE, TYR, TRP |

Unknown residues fall back to polar. The remaining ten entries indicate the
ligand element of the contact: C, N, O, S, P, F, Cl, Br, I, Se (columns 5–14).

Code: `pgl.operator.chemistry_channels`.

## 2. Contact-token statistics (114, columns 15–128)

At the central 5–6 Å window, every contact edge receives a 57-entry token
`φ = [q; s; b; c]`:

| entries | width | content |
|---|---|---|
| `q` | 15 | chemistry channels of the contact |
| `s` | 7 | 4 local heat responses `Σ_k u_ek² exp(−t λ_k)` at `t = 0.25, 1, 4, 16`, and 3 band-pass responses `Σ_k u_ek² (exp(−tλ_k) − exp(−2tλ_k))²` at `t = 0.25, 1, 4` |
| `b` | 32 | interface geometry around the contact (below) |
| `c` | 3 | coupling degree, signed balance and coupling entropy |

The 114 statistics are the element-wise mean and standard deviation of `φ` over
the contact edges, mean first (57 columns) then standard deviation (57 columns).
Complexes with zero contacts at the central window raise instead of producing a
NaN row; batch extraction reports such samples and skips them.

The 32 geometry entries per contact are:

| entries | width | content |
|---|---|---|
| segment length | 1 | `‖ℓ_a − p_a‖` |
| environment counts | 4 | atoms within 4 and 6 Å of the segment midpoint, protein side and ligand side |
| covariance shape | 6 | normalised eigenvalues of the 3×3 covariance of each side within 6 Å |
| radial shells | 12 | Gaussian shells centred at 1–6 Å, separately per side |
| angular moments | 8 | Legendre moments 0–3 of `cos` between the segment axis and the atom offsets, per side |
| imbalance | 1 | `(n6_prot − n6_lig) / (n6_prot + n6_lig)` |

Code: `pgl.features.token_names`, `pgl.features.token_matrix`,
`pgl.features.local_spectral`, `pgl.features.interface_geometry`,
`pgl.features.coupling_summary`.

The token block is the only place where the operator enters the readout: `s`
and `c` are functions of the spectrum and of the coupling matrix of the
persistent operator (see `docs/algorithm.md`).

## 3. All-atom contacts (158, columns 129–286)

This block is independent of the Cα contact topology. Protein heavy atoms
(ATOM records of standard residues, within 8 Å of the ligand) and ligand heavy
atoms are used:

| entries | width | content |
|---|---|---|
| typed counts | 121 | 11 × 11 protein-element × ligand-element contact counts within 4.5 Å, normalised by ligand heavy atoms |
| distance histograms | 20 | 10 bins from 2.0 to 7.0 Å, for the per-ligand-atom and per-protein-atom minimum distances |
| distance summary | 4 | minimum, 5% quantile, median and mean of the per-ligand-atom minimum distance |
| hydrogen bonds | 3 | protein-donor/ligand-acceptor, ligand-donor/protein-acceptor, total, within 4.5 Å |
| aromatic contacts | 2 | aromatic-aromatic atom pairs and the fraction of ligand atoms involved |
| charged contacts | 4 | opposite-charge counts, same-charge count, and the total per ligand atom |
| hydrophobic contacts | 1 | carbon-carbon contacts within 4.0 Å with hydrophobic protein residues |
| interface size | 3 | contact density, number of protein atoms in the shell, ligand heavy-atom count |

The element groups are C, N, O, S, P, F, Cl, Br, I, metal and other. Ligand
functional-group flags come from RDKit SMARTS patterns; for an unsanitized
ligand the group flags stay zero and the rest of the block is still computed.

Code: `pgl.contacts.contact_features`, `pgl.contacts.feature_names`.

## 4. Frozen context (optional, columns 287–1278)

Two frozen encoders provide context. This repository does not run them; it
defines the file interface and the assembly:

| block | columns | file | width |
|---|---|---|---|
| Uni-Mol (ligand) | 287–798 | `ids` + `vectors` | 512 |
| SaProt (protein) | 799–1278 | `ids` + `vectors` | 480 |

```python
from pgl import ContextEmbedding, build_feature_matrix

unimol = ContextEmbedding.load("casf_unimol.npz", "unimol")
saprot = ContextEmbedding.load("casf_saprot.npz", "saprot")
matrix, names, blocks = build_feature_matrix(tokens, ids, {"unimol": unimol, "saprot": saprot})
# matrix.shape == (n, 1279)
```

Rows are aligned by `sample_id`, never by row position. A sample missing one
block keeps NaN in that block instead of an imputed value; XGBoost handles the
missing values natively. With `missing_flags=True` one presence indicator
column per block is appended at the end (`ctx_<tag>_present`), which gives 1,281
columns. The models reported in the paper use the 1,279-column layout.

## 5. Cost

On one CPU core the 287-dimensional block takes about 0.2–0.4 s per complex for
structures of typical size (3000–6000 protein atoms), with no dependence on the
ligand or protein encoders. Context embeddings dominate the reported feature
cost and are produced separately.
