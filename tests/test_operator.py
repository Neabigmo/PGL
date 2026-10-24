"""Tests of the contact topology and the Gauss-linking operator."""

from __future__ import annotations

import numpy as np
import pytest

from pgl.operator import (
    Complex,
    boundary_matrices,
    build_windows,
    gauss_coupling,
    metric_from_coupling,
    parse_complex,
    pgl_operator,
    read_protein_ca,
)


def _tiny_complex() -> Complex:
    """Six vertices: a protein triangle facing a ligand triangle."""
    protein = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 2.2, 0.0]])
    ligand = np.array([[1.3, 0.2, 2.5], [3.4, 0.3, 2.8], [2.0, 1.8, 3.1]])
    coords = np.vstack([protein, ligand])
    births: dict[tuple[int, int], float] = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        births[(i, j)] = 0.0
    for i, j in ((3, 4), (4, 5), (3, 5)):
        births[(i, j)] = 0.0
    cross_edges, cross_coords, cross_ids, chemistry = [], [], [], []
    for i in range(3):
        for j in range(3):
            edge = (i, 3 + j)
            distance = float(np.linalg.norm(coords[i] - coords[3 + j]))
            if distance <= 6.0:
                births[edge] = distance
                cross_edges.append(edge)
                cross_coords.append(coords[list(edge)])
                cross_ids.append(edge)
                channels = np.zeros(15)
                channels[i] = 1.0
                channels[5 + j] = 1.0
                chemistry.append(channels)
    return Complex(
        protein_xyz=protein,
        ligand_xyz=ligand,
        protein_atoms=[],
        ligand=None,
        coords=coords,
        births=births,
        cross_edges=cross_edges,
        cross_coords=np.asarray(cross_coords),
        cross_ids=np.asarray(cross_ids),
        cross_chemistry=np.asarray(chemistry),
    )


def test_boundary_of_boundary_is_zero():
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    b1, b2 = boundary_matrices(4, edges, triangles)
    assert np.allclose(b1 @ b2, 0.0)


def test_boundary_rejects_inconsistent_triangle():
    with pytest.raises(ValueError):
        boundary_matrices(4, [(0, 1), (0, 2), (1, 2)], [(0, 1, 3)])


def test_gauss_coupling_is_symmetric_with_zero_diagonal():
    starts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ends = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ids = np.array([[0, 1], [2, 3], [4, 5]])
    chemistry = np.eye(15)[[0, 1, 2]]
    coupling, signed = gauss_coupling(starts, ends, ids, chemistry)
    assert np.allclose(coupling, coupling.T)
    assert np.allclose(signed, signed.T)
    assert np.allclose(np.diag(coupling), 0.0)


def test_gauss_coupling_is_translation_invariant():
    starts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ends = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    ids = np.array([[0, 1], [2, 3]])
    chemistry = np.eye(15)[[0, 1]]
    coupling, _ = gauss_coupling(starts, ends, ids, chemistry)
    shifted, _ = gauss_coupling(starts + 1000.0, ends + 1000.0, ids, chemistry)
    assert np.allclose(coupling, shifted, atol=1e-9)


def test_gauss_coupling_skips_segments_with_a_shared_endpoint():
    starts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ends = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    ids = np.array([[0, 1], [0, 1]])
    chemistry = np.eye(15)[[0, 1]]
    coupling, _ = gauss_coupling(starts, ends, ids, chemistry)
    assert np.allclose(coupling, 0.0)


def test_metric_is_positive_definite_with_zero_laplacian_rows():
    coupling = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 0.5], [2.0, 0.5, 0.0]])
    metric, laplacian, _ = metric_from_coupling(coupling)
    assert np.allclose(laplacian.sum(axis=1), 0.0)
    assert np.linalg.eigvalsh(metric).min() > 0.0


def test_operator_spectrum_is_nonnegative():
    complex_ = _tiny_complex()
    windows = build_windows(complex_)
    central = windows[1]
    assert central.n_edges > 0
    assert central.eigenvalues.min() >= -1e-8
    assert central.eigenvalues.max() > 0.0


def test_operator_handles_empty_edge_set():
    b1 = np.zeros((3, 0))
    b2p = np.zeros((0, 0))
    operator, eigenvalues, eigenvectors = pgl_operator(b1, b2p, np.eye(0))
    assert operator.shape == (0, 0)
    assert eigenvalues.shape == (0,)


def test_parse_toy_complex(toy_pair):
    complex_ = parse_complex(*toy_pair)
    assert complex_.n_protein > 0
    assert complex_.n_ligand > 0
    assert len(complex_.cross_edges) > 0
    assert complex_.cross_chemistry.shape[1] == 15


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
def test_metric_is_positive_definite_and_bounded(eta):
    """The metric satisfies I <= M <= (1 + eta) I for the coupling Laplacian."""
    rng = np.random.default_rng(3)
    coupling = rng.random((12, 12))
    coupling = np.triu(coupling, k=1)
    coupling = coupling + coupling.T
    metric, laplacian, _ = metric_from_coupling(coupling, eta=eta)
    eigenvalues = np.linalg.eigvalsh(metric)
    assert eigenvalues.min() >= 1.0 - 1e-10
    assert eigenvalues.max() <= 1.0 + eta + 1e-10
    assert np.allclose(laplacian.sum(axis=1), 0.0, atol=1e-12)


def test_read_protein_ca_ignores_hetatm(tmp_path):
    path = tmp_path / "mini.pdb"
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  CA   CA A   2       1.000   0.000   0.000  1.00  0.00          CA\n"
        "ATOM      3  CA  GLY A   3       2.000   0.000   0.000  1.00  0.00           C\n",
        encoding="utf-8",
    )
    atoms = read_protein_ca(path)
    assert len(atoms) == 2
    assert [atom.resname for atom in atoms] == ["ALA", "GLY"]
