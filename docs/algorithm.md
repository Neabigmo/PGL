# Algorithm and code map

This note follows the construction of the PGL operator from the contact graph to
the 287-dimensional contact-token block, and points to the function that
implements each step. Equation labels `eq:segment`, `eq:gauss`, `eq:weight`,
`eq:metric` and `eq:pgl` are the ones used in Section 2 of the manuscript.

## 1. Contact graph and filtration

For one complex, let `P` be the protein Cα atoms within 10 Å of any ligand heavy
atom and `L` the ligand heavy atoms. The vertex set is `P ∪ L`. Three kinds of
edges are created:

* protein Cα pairs closer than 4.5 Å (birth 0),
* bonded ligand atom pairs (birth 0),
* protein-ligand pairs closer than 7 Å (birth = endpoint distance).

The last group is oriented protein→ligand and forms the contact segments
`γ_a(s) = p_a + s(ℓ_a − p_a)`, `s ∈ [0,1]` (`eq:segment`). Three nested
filtration windows, 4–5 Å, 5–6 Å and 6–7 Å, give the inclusion pairs
`(K_4, K_5)`, `(K_5, K_6)` and `(K_6, K_7)`.

Code: `pgl.operator.parse_complex` builds the vertex coordinates, the edge birth
dictionary and the contact segments. `pgl.operator.build_windows` evaluates the
three windows. Parsing rules: `read_protein_ca` keeps ATOM records of standard
residues only (never HETATM, so calcium ions are not mistaken for Cα) and
resolves alternate locations by occupancy, preferring blank/A; `read_ligand`
reads SDF/MOL2 through RDKit and falls back to unsanitized parsing only when
sanitization fails, flagging the fallback on the molecule.

For a window with X-edges `E_X` and Y-edges `E_Y`, the boundary matrices are
`B1_X` (vertices × edges) and `B2_Y` (edges × flag triangles). The persistent
2-chain space is the kernel of the row block `B2_Y[E_Y \ E_X]`; projecting it
onto the X edges gives `B2p = B2_Y[:n_X] Z` with `Z` an orthonormal kernel
basis. `B1_X B2p = 0` and the Schur-complement check
`L_up = Schur(B2_Y B2_Y^T)` are verified numerically and raise on failure.

Code: `boundary_matrices`, `flag_triangles`, `persistent_hodge_laplacian`,
`_persistent_pair`.

## 2. Gauss coupling

The regularized Gauss pairing of two oriented segments is `eq:gauss`:

```
G_ab = (1 / 4π) ∫∫ det(γ̇_a(s), γ̇_b(t), γ_a(s) − γ_b(t))
                / (‖γ_a(s) − γ_b(t)‖² + ε²)^(3/2) ds dt
```

with `ε = 0.5 Å`. The integral is evaluated with a fourth-order Gauss–Legendre
rule on each segment (16 node pairs), so `G_ab` is smooth in the atomic
coordinates. Pairs sharing an endpoint and the diagonal are set to zero.

Each contact carries a 15-dimensional chemistry vector `q_a` (§4). The
similarity weight is `ω_ab = exp(−‖q_a − q_b‖² / 4)` (`eq:weight`) and the
coupling matrix is `W_ab = ω_ab |G_ab|`. The signed matrix `G` is kept because
the signed-balance token reads its row sums.

Code: `pgl.operator.gauss_coupling` returns `(W, G)`; the construction is
covered by the translation-invariance and shared-endpoint tests.

## 3. Metric and persistent operator

With the degree matrix `D = diag(W 1)` and the graph Laplacian `L_G = D − W`,
the metric on the edge space is (`eq:metric`)

```
M = I + η L_G / (2 d_max + δ),    d_max = max_a D_aa,  η = 1,  δ = 0
```

and is symmetric positive definite by construction (checked before use).

The paper writes the operator as the generalized problem (`eq:pgl`)

```
(B1_X^T B1_X + M B2p B2p^T M) v = λ M v.
```

The implementation solves the symmetric representative

```
L = M^(-1/2) B1_X^T B1_X M^(-1/2) + M^(1/2) B2p B2p^T M^(1/2),
```

which has the same eigenvalues and is positive semidefinite in the standard
inner product, so ordinary `eigh` gives real nonnegative eigenvalues
`(λ_k, u_k)`. The zero eigenvalue multiplicity equals the first persistent
Betti number of the pair; `B2p = 0` reduces the operator to the persistent edge
Laplacian.

Code: `pgl.operator.metric_from_coupling`, `pgl.operator.pgl_operator`,
`pgl.operator.build_window`. All linear algebra is float64.

## 4. Contact tokens and the 287-dimensional block

At the central 5–6 Å window, each contact edge `a` receives 57 token entries
`φ_a = [q_a; s_a; b_a]`:

| part | dim | content |
|---|---|---|
| `q_a` | 15 | 5 protein residue classes + 10 ligand element indicators |
| `s_a` | 7 | 4 heat responses `Σ_k u_ak² e^(−t λ_k)`, `t ∈ {0.25, 1, 4, 16}` and 3 band-pass responses at `t ∈ {0.25, 1, 4}` |
| `b_a` | 32 | segment length, environment counts and covariance shape of both sides within 4 and 6 Å, radial shells 1–6 Å, angular moments up to order 3, side imbalance |
| coupling | 3 | weighted degree, signed balance, coupling-profile entropy |

Aggregating over the `m` contacts gives the blocks used by the readout:

```
chem15    = mean_a q_a                                     (15)
token114  = [mean_a φ_a ; std_a φ_a]                       (114)
atomic158 = typed all-atom contact block of the interface   (158)
```

For a complex, the readout input is `[chem15; token114; atomic158]`, optionally
followed by the context blocks (see `docs/features.md`).

Code: `pgl.features.local_spectral`, `pgl.features.interface_geometry`,
`pgl.features.coupling_summary`, `pgl.features.token_matrix`,
`pgl.features.contact_features`, `pgl.contacts.contact_features`.

## 5. Numerical conventions

* `ε = 0.5 Å`, Gauss–Legendre order 4, `η = 1`.
* Filtration windows `(4,5)`, `(5,6)`, `(6,7)` Å; contact tokens use the middle
  window, where contacts are densest.
* `d_max` is the largest weighted degree of the coupling matrix of the window.
* `eigh` is applied to explicitly symmetrised matrices; a negative eigenvalue
  below `−1e-8` is treated as a numerical failure.
* Feature extraction is deterministic: repeated calls on the same structures
  return identical vectors.

## 6. Readout

The readout is an XGBoost regressor with one predeclared hyper-parameter set
(708 trees, depth 6, learning rate 0.0529, subsample 0.7407,
`colsample_bytree` 0.9009, `min_child_weight` 7.087, `reg_lambda` 7.479,
`reg_alpha` 0.281, squared error, histogram tree method). Three fixed seeds are
reported: 20261022, 20261023 and 20261024. The hyper-parameters were fixed
before the benchmark runs; no selection is performed per dataset.

Code: `pgl.readout.XGBReadout` and `pgl.readout.XGB_PRESET`.
