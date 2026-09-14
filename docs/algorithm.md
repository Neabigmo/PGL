# Algorithm

The implementation is extracted from the project canonical numerical core and parser. Protein Cα and ligand heavy-atom vertices are used; ligand bonds, local protein edges, and protein–ligand cross contacts form nested graph filtrations. Cross edges are oriented protein→ligand. The 4→5, 5→6, and 6→7 Å windows define persistent upper boundaries.

For `B2Y=[BX; Bout]`, an orthonormal basis `Z` of `ker(Bout)` gives `B2XY=BX Z`. The regularized Gauss relation is integrated with fourth-order Gauss–Legendre quadrature and ε=0.5 Å. Shared-endpoint pairs and the diagonal are zeroed. Absolute Gauss values are chemistry-weighted by `exp(-||qa-qb||²/4)`, yielding `MG=I+LG/(2 dmax+1e-12)`. The symmetric generalized operator is solved in float64 and normalized by its positive-eigenvalue mean.
