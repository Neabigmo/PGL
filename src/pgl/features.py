"""The 287-dimensional contact-token representation.

A complex is summarised by three blocks that share one definition across
datasets:

===============  ====  =====================================================
block            dim   content
===============  ====  =====================================================
chemistry         15   mean residue/element channels over the contact edges
token statistics 114   mean and standard deviation of the 57 contact tokens
atomic contacts  158   all-atom typed contact counts and distance summaries
===============  ====  =====================================================

The contact tokens are the per-edge readout of the operator. Each token has
57 entries: 15 chemistry channels, 7 local spectral responses, 32 interface
geometry descriptors and 3 coupling summaries. Averaging and taking the
standard deviation over the contact edges gives the 114-dimensional block.

The representation is label-free and fit-free: two calls on the same
complex return bit-identical vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import contacts
from .operator import (
    BAND_TIMES,
    HEAT_TIMES,
    Complex,
    Window,
    parse_complex,
)

CHEMISTRY_DIM = 15
TOKEN_DIM = 57
TOKEN_STAT_DIM = 2 * TOKEN_DIM
ATOMIC_DIM = 158
FEATURE_DIM = CHEMISTRY_DIM + TOKEN_STAT_DIM + ATOMIC_DIM  # 287

BLOCK_SLICES = {
    "chemistry": (0, CHEMISTRY_DIM),
    "token_statistics": (CHEMISTRY_DIM, CHEMISTRY_DIM + TOKEN_STAT_DIM),
    "atomic_contacts": (
        CHEMISTRY_DIM + TOKEN_STAT_DIM,
        CHEMISTRY_DIM + TOKEN_STAT_DIM + ATOMIC_DIM,
    ),
}

CHEMISTRY_GROUPS = ("hydrophobic", "polar", "positive", "negative", "aromatic")
LIGAND_ELEMENTS = ("C", "N", "O", "S", "P", "F", "CL", "BR", "I", "SE")


@dataclass
class ContactTokens:
    """Contact-token representation of one complex."""

    vector: np.ndarray
    token_matrix: np.ndarray
    counts: dict[str, float] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])


def chemistry_abundance(complex_: Complex) -> np.ndarray:
    """Mean chemistry channel over the contact edges (15 dimensions)."""
    if len(complex_.cross_chemistry) == 0:
        return np.zeros(CHEMISTRY_DIM)
    return complex_.cross_chemistry.mean(axis=0)


def local_spectral(window: Window) -> np.ndarray:
    """Local heat and band-pass response of the operator at every edge.

    The response of edge ``e`` is ``sum_k u_ek^2 exp(-t lambda_k)``. Heat times
    probe different spectral scales; the band-pass terms isolate the scale
    around each time. Returns an ``(n_edges, 7)`` array.
    """
    squared = window.eigenvectors**2
    eigenvalues = window.eigenvalues
    responses = [squared @ np.exp(-time * eigenvalues) for time in HEAT_TIMES]
    responses += [
        squared @ (np.exp(-time * eigenvalues) - np.exp(-2.0 * time * eigenvalues)) ** 2
        for time in BAND_TIMES
    ]
    return np.stack(responses, axis=1)


def interface_geometry(
    window: Window, protein_xyz: np.ndarray, ligand_xyz: np.ndarray
) -> np.ndarray:
    """Geometry of the environment around each contact segment (32 dimensions).

    For each contact the function describes the shape and radial/angular
    distribution of the protein atoms and of the ligand atoms within 6 A of the
    segment midpoint, and the imbalance between the two sides.
    """
    protein_side = window.cross_coords[:, :3]
    ligand_side = window.cross_coords[:, 3:]
    midpoint = (protein_side + ligand_side) / 2.0
    axis = ligand_side - protein_side
    length = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = axis / (length + 1e-8)

    def side(xyz: np.ndarray):
        delta = xyz[None, :, :] - midpoint[:, None, :]
        distance = np.linalg.norm(delta, axis=2)
        inside = (distance <= 6.0).astype(np.float64)
        n4 = (distance <= 4.0).sum(1, keepdims=True)
        n6 = inside.sum(1, keepdims=True)
        denominator = np.maximum(n6, 1)
        covariance = np.einsum("eni,enj,en->eij", delta, delta, inside) / denominator[:, :, None]
        values = np.linalg.eigvalsh(covariance)[:, ::-1].clip(0)
        shape = np.c_[
            (values[:, 0] - values[:, 1]) / (values[:, 0] + 1e-8),
            (values[:, 1] - values[:, 2]) / (values[:, 0] + 1e-8),
            values[:, 2] / (values[:, 0] + 1e-8),
        ]
        centers = np.arange(1, 7)
        radial = np.exp(-0.5 * ((distance[:, :, None] - centers) / 0.75) ** 2)
        radial = (radial * inside[:, :, None]).sum(1) / denominator
        unit = delta / (distance[:, :, None] + 1e-8)
        cosine = np.clip((unit * axis[:, None, :]).sum(2), -1.0, 1.0)
        angular = np.stack(
            [
                cosine,
                0.5 * (3 * cosine * cosine - 1),
                0.5 * (5 * cosine**3 - 3 * cosine),
                (35 * cosine**4 - 30 * cosine * cosine + 3) / 8,
            ],
            axis=-1,
        )
        angular = (angular * inside[:, :, None]).sum(1) / denominator
        return n4, n6, shape, radial, angular

    protein_n4, protein_n6, protein_shape, protein_radial, protein_angular = side(protein_xyz)
    ligand_n4, ligand_n6, ligand_shape, ligand_radial, ligand_angular = side(ligand_xyz)
    return np.c_[
        length,
        protein_n4,
        ligand_n4,
        protein_n6,
        ligand_n6,
        protein_shape,
        ligand_shape,
        protein_radial,
        ligand_radial,
        protein_angular,
        ligand_angular,
        (protein_n6 - ligand_n6) / (protein_n6 + ligand_n6 + 1e-8),
    ]


def coupling_summary(window: Window) -> np.ndarray:
    """Coupling summaries of each contact segment (3 dimensions).

    The three entries are the weighted degree, the signed balance and the
    entropy of the normalised coupling profile. They are evaluated on the
    contact segments only.
    """
    degree = window.coupling.sum(axis=1)
    signed = window.signed_coupling.sum(axis=1)
    normalized = window.coupling / (degree[:, None] + 1e-8)
    entropy = -(normalized * np.log(normalized + 1e-8)).sum(axis=1)
    summary = np.c_[degree, signed / (degree + 1e-8), entropy]
    return summary[window.cross_rows]


def token_matrix(complex_: Complex, windows: list[Window]) -> np.ndarray:
    """Per-edge contact tokens at the central (5-6 A) window, shape (m, 57)."""
    window = windows[1]
    if len(window.cross_rows) == 0:
        raise ValueError(
            "no protein-ligand contact in the central 5-6 A window; "
            "the contact-token block is undefined for this complex"
        )
    local = np.c_[
        window.chemistry,
        local_spectral(window)[window.cross_rows],
        interface_geometry(window, complex_.protein_xyz, complex_.ligand_xyz),
        coupling_summary(window),
    ]
    if local.shape[1] != TOKEN_DIM:
        raise RuntimeError(f"token width {local.shape[1]}, expected {TOKEN_DIM}")
    return local


def contact_features(
    protein_path,
    ligand_path,
    complex_: Complex | None = None,
    windows: list[Window] | None = None,
) -> ContactTokens:
    """Compute the 287-dimensional contact-token representation.

    Parameters
    ----------
    protein_path:
        Protein structure (PDB). The full chain is used for the all-atom
        contact block.
    ligand_path:
        Ligand structure (SDF or MOL2).
    complex_:
        Optional pre-parsed complex, to avoid parsing twice.
    windows:
        Optional pre-built windows.

    Returns
    -------
    ContactTokens
        ``vector`` has shape ``(287,)`` and ``token_matrix`` has shape
        ``(n_contacts, 57)``.
    """
    if complex_ is None:
        complex_ = parse_complex(protein_path, ligand_path)
    if windows is None:
        from .operator import build_windows

        windows = build_windows(complex_)
    tokens = token_matrix(complex_, windows)
    statistics = np.r_[tokens.mean(axis=0), tokens.std(axis=0)]
    vector = np.r_[
        chemistry_abundance(complex_),
        statistics,
        contacts.contact_features(protein_path, ligand_path),
    ]
    if vector.shape != (FEATURE_DIM,):
        raise RuntimeError(f"contact-token width {vector.shape}, expected ({FEATURE_DIM},)")
    return ContactTokens(
        vector=vector.astype(np.float32),
        token_matrix=tokens,
        counts={
            "n_protein": complex_.n_protein,
            "n_ligand": complex_.n_ligand,
            "n_contacts": len(tokens),
            "n_edges": [float(window.n_edges) for window in windows],
            "betti1": [float(window.betti1) for window in windows],
            "chain_error": [window.chain_error for window in windows],
            "schur_error": [window.schur_error for window in windows],
        },
    )


def token_names() -> list[str]:
    """Names of the 57 contact-token entries."""
    names = [f"token_chem_{group}" for group in CHEMISTRY_GROUPS]
    names += [f"token_lig_{element}" for element in LIGAND_ELEMENTS]
    names += [f"token_heat_{time:g}" for time in HEAT_TIMES]
    names += [f"token_band_{time:g}" for time in BAND_TIMES]
    names += ["token_geom_span"]
    names += ["token_geom_prot_n4", "token_geom_lig_n4", "token_geom_prot_n6", "token_geom_lig_n6"]
    names += [f"token_geom_prot_shape_{i}" for i in range(3)]
    names += [f"token_geom_lig_shape_{i}" for i in range(3)]
    names += [f"token_geom_prot_radial_{i}" for i in range(6)]
    names += [f"token_geom_lig_radial_{i}" for i in range(6)]
    names += [f"token_geom_prot_angular_{i}" for i in range(4)]
    names += [f"token_geom_lig_angular_{i}" for i in range(4)]
    names += ["token_geom_balance"]
    names += ["token_coupling_degree", "token_coupling_balance", "token_coupling_entropy"]
    if len(names) != TOKEN_DIM:
        raise RuntimeError(f"token name list has {len(names)} entries, expected {TOKEN_DIM}")
    return names


def feature_names() -> list[str]:
    """Names of the 287 contact-token features, in block order."""
    names = [f"chem_{group}" for group in CHEMISTRY_GROUPS]
    names += [f"chem_lig_{element}" for element in LIGAND_ELEMENTS]
    names += [f"token_mean_{name}" for name in token_names()]
    names += [f"token_std_{name}" for name in token_names()]
    names += contacts.feature_names()
    if len(names) != FEATURE_DIM:
        raise RuntimeError(f"feature name list has {len(names)} entries, expected {FEATURE_DIM}")
    return names


__all__ = [
    "ATOMIC_DIM",
    "BLOCK_SLICES",
    "CHEMISTRY_DIM",
    "ContactTokens",
    "FEATURE_DIM",
    "TOKEN_DIM",
    "TOKEN_STAT_DIM",
    "chemistry_abundance",
    "contact_features",
    "coupling_summary",
    "feature_names",
    "interface_geometry",
    "local_spectral",
    "token_matrix",
    "token_names",
]
