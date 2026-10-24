"""Contact topology, structure parsing and the persistent Gauss-linking operator.

The operator is built on the contact graph of one protein-ligand complex.
Protein C-alpha atoms and ligand heavy atoms are vertices. A vertex pair
becomes an edge once its distance falls below a filtration radius, and every
protein-ligand pair within the cross cutoff is an oriented contact segment.
The Gauss linking number of two contact segments defines a signed coupling;
its absolute value defines the weighted adjacency used by the operator.

Only numpy and scipy are required for the operator itself. RDKit is imported
lazily by :func:`read_ligand` to parse ligand files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

# Filtration windows (r, s): edges with birth <= r form the X graph, edges with
# birth <= s form the Y graph. The three windows are nested.
WINDOWS = ((4.0, 5.0), (5.0, 6.0), (6.0, 7.0))

# Heat times of the spectral response and of its band-pass variant.
HEAT_TIMES = (0.25, 1.0, 4.0, 16.0)
BAND_TIMES = (0.25, 1.0, 4.0)

# Protein-ligand contacts enter the cross graph below this distance.
CROSS_CUTOFF = 7.0

STANDARD_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "MSE",
    "SEC",
    "PYL",
    "HID",
    "HIE",
    "HIP",
    "CYX",
    "ASH",
    "GLH",
    "LYN",
}

# Residue classes used by the 15 chemistry channels.
HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PRO", "GLY"}
POLAR = {"SER", "THR", "ASN", "GLN", "CYS"}
POSITIVE = {"LYS", "ARG", "HIS"}
NEGATIVE = {"ASP", "GLU"}
AROMATIC = {"PHE", "TYR", "TRP"}


@dataclass(frozen=True)
class ProteinAtom:
    """One protein atom selected for the contact graph."""

    xyz: np.ndarray
    resname: str
    chain: str
    resseq: str
    icode: str
    name: str = ""
    element: str = "C"


@dataclass(frozen=True)
class Ligand:
    """Ligand heavy atoms, bonds and the parsed RDKit molecule."""

    xyz: np.ndarray
    bonds: list[tuple[int, int]]
    elements: list[str]
    mol: object
    sanitized: bool


@dataclass
class Complex:
    """Parsed protein-ligand complex with the contact graph inputs."""

    protein_xyz: np.ndarray  # retained C-alpha coordinates
    ligand_xyz: np.ndarray  # ligand heavy-atom coordinates
    protein_atoms: list[ProteinAtom]  # parsed protein atoms, same order as protein_xyz
    ligand: Ligand  # full ligand record
    coords: np.ndarray  # protein C-alpha followed by ligand heavy atoms
    births: dict[tuple[int, int], float]  # edge -> filtration birth distance
    cross_edges: list[tuple[int, int]]  # oriented protein -> ligand contacts
    cross_coords: np.ndarray  # (m, 6) endpoint coordinates of the contacts
    cross_ids: np.ndarray  # (m, 2) global vertex ids of the contacts
    cross_chemistry: np.ndarray  # (m, 15) chemistry channels of the contacts

    @property
    def n_protein(self) -> int:
        return len(self.protein_xyz)

    @property
    def n_ligand(self) -> int:
        return len(self.ligand_xyz)


@dataclass
class Window:
    """One filtration window with its persistent topology and Gauss coupling."""

    r: float  # radius of the X graph
    s: float  # radius of the Y graph
    coords: np.ndarray  # vertex coordinates of the complex
    edges_x: list[tuple[int, int]]  # edges of the X graph
    edges_y: list[tuple[int, int]]  # edges of the Y graph
    triangles_y: list[tuple[int, int, int]]  # flag triangles of the Y graph
    b1x: np.ndarray  # vertex-edge boundary of X
    b2p: np.ndarray  # persistent upper boundary
    betti1: int  # first persistent Betti number of the pair
    cross_rows: np.ndarray  # rows of the contacts inside the X edge set
    cross_coords: np.ndarray  # endpoint coordinates of those contacts
    cross_ids: np.ndarray  # endpoint vertex ids of those contacts
    chemistry: np.ndarray  # (m, 15) chemistry channels of those contacts
    coupling: np.ndarray  # weighted coupling over the X edges
    signed_coupling: np.ndarray  # signed coupling over the X edges
    metric: np.ndarray  # metric on the X edge space
    laplacian: np.ndarray  # coupling graph Laplacian
    eigenvalues: np.ndarray  # operator spectrum
    eigenvectors: np.ndarray  # operator eigenvectors, one per column
    chain_error: float  # residual of B1_X @ B2p
    schur_error: float  # difference between the Schur complement and the persistent operator

    @property
    def n_edges(self) -> int:
        return len(self.edges_x)


# --------------------------------------------------------------------------- #
# boundary operators and persistence
# --------------------------------------------------------------------------- #
def boundary_matrices(
    n_vertices: int,
    edges: list[tuple[int, int]],
    triangles: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the vertex-edge (B1) and edge-triangle (B2) boundary matrices.

    ``B1 @ B2 == 0`` is checked exactly; a violation means the triangle list is
    inconsistent with the edge list and is treated as an error.
    """
    edges = [tuple(sorted(map(int, e))) for e in edges]
    if len(edges) != len(set(edges)) or any(a < 0 or b >= n_vertices or a >= b for a, b in edges):
        raise ValueError("invalid or duplicated edge")
    index = {e: i for i, e in enumerate(edges)}
    b1 = np.zeros((n_vertices, len(edges)), dtype=np.float64)
    for j, (a, b) in enumerate(edges):
        b1[a, j] = -1.0
        b1[b, j] = 1.0
    triangles = [tuple(sorted(map(int, t))) for t in triangles]
    if len(triangles) != len(set(triangles)) or any(len(t) != 3 for t in triangles):
        raise ValueError("invalid or duplicated triangle")
    b2 = np.zeros((len(edges), len(triangles)), dtype=np.float64)
    for j, (a, b, c) in enumerate(triangles):
        for edge, sign in (((b, c), 1.0), ((a, c), -1.0), ((a, b), 1.0)):
            if edge not in index:
                raise ValueError("triangle has a missing face")
            b2[index[edge], j] = sign
    error = float(np.max(np.abs(b1 @ b2))) if b1.size and b2.size else 0.0
    if error != 0.0:
        raise ValueError(f"boundary of boundary is nonzero: {error}")
    return b1, b2


def flag_triangles(
    n_vertices: int, edges: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Return sorted unique edges and the flag (clique) triangles they span."""
    unique = sorted({tuple(sorted(map(int, e))) for e in edges})
    adjacency = [set() for _ in range(n_vertices)]
    for a, b in unique:
        if a == b or a < 0 or b >= n_vertices:
            raise ValueError("invalid edge")
        adjacency[a].add(b)
        adjacency[b].add(a)
    triangles = []
    for a, b in unique:
        triangles.extend((a, b, c) for c in sorted(adjacency[a] & adjacency[b]) if c > b)
    return unique, triangles


def _rank(matrix: np.ndarray, rtol: float = 1e-10) -> int:
    if 0 in matrix.shape:
        return 0
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular > rtol * max(float(singular[0]), 1e-300)))


def persistent_hodge_laplacian(b2y: np.ndarray, n_edges_x: int, rtol: float = 1e-10):
    """Persistent upper Hodge Laplacian and its persistent 2-chain basis.

    The 2-chains of Y that are killed by the boundary of the Y-only edges are
    projected onto the X edges; the result is the upper operator of the pair
    (X, Y).
    """
    b2y = np.asarray(b2y, dtype=np.float64)
    if b2y.ndim != 2 or not 0 <= int(n_edges_x) <= b2y.shape[0]:
        raise ValueError("invalid B2Y or lower-edge count")
    nx = int(n_edges_x)
    boundary_out = b2y[nx:]
    if boundary_out.shape[0] == 0:
        kernel = np.eye(b2y.shape[1], dtype=np.float64)
    else:
        kernel = null_space(boundary_out, rcond=rtol)
    b2p = b2y[:nx] @ kernel
    hup = b2p @ b2p.T
    return (hup + hup.T) / 2.0, b2p


def _persistent_pair(n_vertices: int, births: dict[tuple[int, int], float], r: float, s: float):
    if r > s:
        raise ValueError("window must satisfy r <= s")
    edges_x, _ = flag_triangles(n_vertices, [e for e, birth in births.items() if birth <= r])
    edges_y, triangles_y = flag_triangles(
        n_vertices, [e for e, birth in births.items() if birth <= s]
    )
    edge_set = set(edges_x)
    order = edges_x + [e for e in edges_y if e not in edge_set]
    b1x, _ = boundary_matrices(n_vertices, edges_x, [])
    b1y, b2y = boundary_matrices(n_vertices, order, triangles_y)
    n_edges_x = len(edges_x)
    hup, b2p = persistent_hodge_laplacian(b2y, n_edges_x)
    chain_error = float(np.max(np.abs(b1x @ b2p))) if b1x.size and b2p.size else 0.0
    if chain_error > 1e-8:
        raise ValueError(f"persistent chain law failed: {chain_error}")
    h = b2y @ b2y.T
    a = h[:n_edges_x, :n_edges_x]
    b = h[:n_edges_x, n_edges_x:]
    c = h[n_edges_x:, n_edges_x:]
    schur = a if c.size == 0 else a - b @ np.linalg.pinv(c, rcond=1e-10) @ b.T
    schur = (schur + schur.T) / 2.0
    schur_error = float(np.max(np.abs(schur - hup))) if n_edges_x else 0.0
    betti1 = int(n_edges_x - _rank(b1x) - _rank(b2p))
    return edges_x, order, triangles_y, b1x, b1y, b2y, b2p, betti1, chain_error, schur_error


# --------------------------------------------------------------------------- #
# Gauss coupling between contact segments
# --------------------------------------------------------------------------- #
def gauss_coupling(
    starts: np.ndarray,
    ends: np.ndarray,
    endpoint_ids: np.ndarray,
    chemistry: np.ndarray,
    epsilon: float = 0.5,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss linking of every pair of contact segments.

    The double integral is evaluated with a fourth-order Gauss-Legendre rule on
    each segment, so the coupling is smooth in the endpoint coordinates. The
    result is weighted by the chemistry agreement of the two segments and is
    returned both as a nonnegative adjacency and as a signed matrix.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    ids = np.asarray(endpoint_ids, dtype=np.int64)
    q = np.asarray(chemistry, dtype=np.float64)
    m = len(starts)
    if starts.shape != (m, 3) or ends.shape != (m, 3) or ids.shape != (m, 2) or q.shape != (m, 15):
        raise ValueError("inconsistent Gauss inputs")
    if m == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))

    nodes, weights = np.polynomial.legendre.leggauss(order)
    points = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    direction = ends - starts
    signed = np.zeros((m, m), dtype=np.float64)
    for i, j in combinations(range(m), 2):
        if set(ids[i]).intersection(ids[j]):
            continue
        numerator = float(np.dot(starts[i] - starts[j], np.cross(direction[i], direction[j])))
        integral = 0.0
        for a, wa in zip(points, weights, strict=True):
            for b, wb in zip(points, weights, strict=True):
                d = starts[i] + a * direction[i] - starts[j] - b * direction[j]
                integral += wa * wb / (np.dot(d, d) + epsilon**2) ** 1.5
        signed[i, j] = signed[j, i] = numerator * integral / (4 * math.pi)
    sq_distance = np.sum((q[:, None, :] - q[None, :, :]) ** 2, axis=-1)
    signed *= np.exp(-sq_distance / 4.0)
    np.fill_diagonal(signed, 0.0)
    return np.abs(signed), signed


def metric_from_coupling(coupling: np.ndarray, eta: float = 1.0):
    """Positive-definite metric on the edge space induced by the coupling.

    ``M = I + eta * L_G / (2 d_max)`` where ``L_G = D - W`` is the weighted
    graph Laplacian of the coupling. Normalising by the largest degree keeps
    the spectrum of ``M`` close to one for every complex size.
    """
    coupling = np.asarray(coupling, dtype=np.float64)
    degree = coupling.sum(axis=1)
    laplacian = np.diag(degree) - coupling
    dmax = float(degree.max()) if len(degree) else 0.0
    if dmax <= 0:
        return np.eye(len(coupling)), laplacian, 0.0
    metric = np.eye(len(coupling)) + eta * laplacian / (2.0 * dmax + 1e-12)
    return metric, laplacian, dmax


def pgl_operator(b1x: np.ndarray, b2p: np.ndarray, metric: np.ndarray):
    """Assemble the persistent Gauss-linking operator and diagonalise it.

    ``L = M^{-1/2} B1^T B1 M^{-1/2} + M^{1/2} B2p B2p^T M^{1/2}``.

    The first term is the persistent edge Laplacian, the second carries the
    Gauss-weighted two-chain coupling. The operator is symmetric and positive
    semidefinite.
    """
    if b1x.shape[1] == 0:
        return np.zeros((0, 0)), np.zeros(0), np.zeros((0, 0))
    values, vectors = np.linalg.eigh((metric + metric.T) / 2.0)
    if float(values.min()) <= 0:
        raise ValueError("metric is not positive definite")
    sqrt_metric = (vectors * np.sqrt(values)) @ vectors.T
    inverse_sqrt_metric = (vectors / np.sqrt(values)) @ vectors.T
    h = b2p @ b2p.T
    operator = (
        inverse_sqrt_metric @ (b1x.T @ b1x) @ inverse_sqrt_metric + sqrt_metric @ h @ sqrt_metric
    )
    operator = (operator + operator.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    if eigenvalues.size and eigenvalues.min() < -1e-8:
        raise ValueError("negative PGL spectrum")
    return operator, np.maximum(eigenvalues, 0.0), eigenvectors


def build_window(
    complex_: Complex,
    r: float,
    s: float,
    eta: float = 1.0,
) -> Window:
    """Build the persistent topology and the operator for one window."""
    n = len(complex_.coords)
    (
        edges_x,
        edges_y,
        triangles_y,
        b1x,
        _b1y,
        _b2y,
        b2p,
        betti1,
        chain_error,
        schur_error,
    ) = _persistent_pair(n, complex_.births, r, s)

    row = {edge: i for i, edge in enumerate(edges_x)}
    cross_rows = np.asarray([row[e] for e in complex_.cross_edges if e in row], dtype=int)
    cross_coords = np.asarray(
        [complex_.cross_coords[i] for i, e in enumerate(complex_.cross_edges) if e in row],
        dtype=np.float64,
    ).reshape((-1, 6))
    cross_ids = np.asarray(
        [complex_.cross_ids[i] for i, e in enumerate(complex_.cross_edges) if e in row],
        dtype=np.int64,
    ).reshape((-1, 2))
    cross_chemistry = np.asarray(
        [complex_.cross_chemistry[i] for i, e in enumerate(complex_.cross_edges) if e in row],
        dtype=np.float64,
    ).reshape((-1, 15))

    coupling, signed = gauss_coupling(
        cross_coords[:, :3], cross_coords[:, 3:], cross_ids, cross_chemistry
    )
    coupling_all = np.zeros((len(edges_x), len(edges_x)), dtype=np.float64)
    signed_all = np.zeros((len(edges_x), len(edges_x)), dtype=np.float64)
    if len(cross_rows):
        coupling_all[np.ix_(cross_rows, cross_rows)] = coupling
        signed_all[np.ix_(cross_rows, cross_rows)] = signed

    if coupling_all.size:
        metric, laplacian, _dmax = metric_from_coupling(coupling_all, eta)
    else:
        metric = np.eye(len(edges_x))
        laplacian = np.zeros((len(edges_x), len(edges_x)))
    _operator, eigenvalues, eigenvectors = pgl_operator(b1x, b2p, metric)
    return Window(
        r=r,
        s=s,
        coords=complex_.coords,
        edges_x=edges_x,
        edges_y=edges_y,
        triangles_y=triangles_y,
        b1x=b1x,
        b2p=b2p,
        betti1=betti1,
        cross_rows=cross_rows,
        cross_coords=cross_coords,
        cross_ids=cross_ids,
        chemistry=cross_chemistry,
        coupling=coupling_all,
        signed_coupling=signed_all,
        metric=metric,
        laplacian=laplacian,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        chain_error=chain_error,
        schur_error=schur_error,
    )


def build_windows(complex_: Complex, eta: float = 1.0) -> list[Window]:
    """Build the three nested filtration windows."""
    return [build_window(complex_, r, s, eta=eta) for r, s in WINDOWS]


# --------------------------------------------------------------------------- #
# structure parsing
# --------------------------------------------------------------------------- #
def _element_from_line(line: str, atom_name: str) -> str:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element
    name = atom_name.strip()
    if not name:
        return "X"
    two_letter = name[:2].upper()
    if (
        two_letter
        in {
            "CL",
            "BR",
            "NA",
            "MG",
            "ZN",
            "FE",
            "CA",
            "MN",
            "CU",
            "NI",
            "CO",
            "CD",
            "HG",
            "SE",
        }
        and len(name) >= 2
    ):
        if not name[0].isalpha() or name[0].isupper():
            return two_letter
    return name[0].upper()


def read_protein_ca(path: str | Path) -> list[ProteinAtom]:
    """Parse protein C-alpha atoms from ATOM records.

    Only standard residues are kept, HETATM records (including calcium ions)
    are skipped, and alternate locations are resolved by occupancy with a
    preference for blank/A.
    """
    best: dict[tuple, tuple] = {}
    order: list[tuple] = []
    with open(path, errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[13:14] == "H":
                continue
            resname = line[17:20].strip().upper()
            if resname not in STANDARD_AA:
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            chain = line[21:22]
            resseq = line[22:26].strip()
            icode = line[26:27]
            altloc = line[16:17]
            try:
                occupancy = float(line[54:60])
            except ValueError:
                occupancy = 1.0
            key = (chain, resseq, icode)
            if key not in best:
                best[key] = (occupancy, altloc, xyz, resname, chain)
                order.append(key)
            else:
                current = best[key]
                better = occupancy > current[0] or (
                    occupancy == current[0]
                    and altloc in (" ", "A")
                    and current[1] not in (" ", "A")
                )
                if better:
                    best[key] = (occupancy, altloc, xyz, resname, chain)
    atoms = [
        ProteinAtom(
            xyz=np.asarray(best[key][2], dtype=np.float64),
            resname=best[key][3],
            chain=best[key][4],
            resseq=key[1],
            icode=key[2],
            name="CA",
            element="C",
        )
        for key in order
    ]
    if not atoms:
        raise ValueError(f"no protein C-alpha atoms in {path}")
    return atoms


def read_protein_heavy(path: str | Path, max_atoms: int = 20000) -> list[ProteinAtom]:
    """Parse protein heavy atoms from ATOM records of standard residues."""
    atoms: list[ProteinAtom] = []
    seen: set[tuple] = set()
    with open(path, errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            if resname not in STANDARD_AA:
                continue
            if line[16:17] not in (" ", "A"):
                continue
            name = line[12:16].strip().upper()
            element = _element_from_line(line, name)
            if element == "H":
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            key = (line[21:22], line[22:26].strip(), line[26:27], name)
            if key in seen:
                continue
            seen.add(key)
            atoms.append(
                ProteinAtom(
                    xyz=np.asarray(xyz, dtype=np.float64),
                    resname=resname,
                    chain=line[21:22],
                    resseq=line[22:26].strip(),
                    icode=line[26:27],
                    name=name,
                    element=element,
                )
            )
            if len(atoms) >= max_atoms:
                break
    return atoms


def read_ligand(path: str | Path) -> Ligand:
    """Read ligand heavy atoms, bonds and elements with coordinates.

    Sanitized parsing is preferred. When sanitization fails, the molecule is
    read without sanitization and flagged, so chemistry-derived features can
    be recognised as unavailable by :mod:`pgl.contacts`.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    path = str(path)
    mol = None
    sanitized = True
    if path.lower().endswith((".sdf", ".mol")):
        mol = Chem.MolFromMolFile(path, sanitize=True, removeHs=False)
        if mol is None:
            mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:  # noqa: BLE001 - malformed chemistry is not fatal here
                    mol = None
    if mol is None and path.lower().endswith(".mol2"):
        mol = Chem.MolFromMol2File(path, sanitize=True, removeHs=False)
        if mol is None:
            mol = Chem.MolFromMol2File(path, sanitize=False, removeHs=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:  # noqa: BLE001
                    mol = None
    if mol is None:
        sanitized = False
        try:
            if path.lower().endswith((".sdf", ".mol")):
                mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=False)
            elif path.lower().endswith(".mol2"):
                mol = Chem.MolFromMol2File(path, sanitize=False, removeHs=False)
        except Exception:  # noqa: BLE001
            mol = None
        if mol is not None:
            try:
                mol.UpdatePropertyCache(strict=False)
            except Exception:  # noqa: BLE001
                pass
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"ligand cannot be read with coordinates: {path}")
    try:
        mol.SetProp("pgl_ligand_sanitized", "1" if sanitized else "0")
    except Exception:  # noqa: BLE001
        pass
    conformer = mol.GetConformer()
    coordinates = np.asarray(
        [
            [
                conformer.GetAtomPosition(i).x,
                conformer.GetAtomPosition(i).y,
                conformer.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ]
    )
    heavy = [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetAtomicNum() > 1]
    if not heavy:
        raise ValueError("ligand has no heavy atoms")
    renumber = {old: new for new, old in enumerate(heavy)}
    bonds = [
        (renumber[b.GetBeginAtomIdx()], renumber[b.GetEndAtomIdx()])
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in renumber and b.GetEndAtomIdx() in renumber
    ]
    elements = [mol.GetAtomWithIdx(i).GetSymbol().upper() for i in heavy]
    return Ligand(
        xyz=coordinates[heavy],
        bonds=bonds,
        elements=elements,
        mol=mol,
        sanitized=sanitized,
    )


def chemistry_channels(resname: str, element: str) -> np.ndarray:
    """Return the 15 chemistry channels of one protein-ligand atom pair.

    The first five entries encode the residue class of the protein atom
    (hydrophobic, polar, positive, negative, aromatic); the remaining ten
    encode the ligand element.
    """
    residue = resname.upper()
    if residue in HYDROPHOBIC:
        group = 0
    elif residue in POLAR:
        group = 1
    elif residue in POSITIVE:
        group = 2
    elif residue in NEGATIVE:
        group = 3
    elif residue in AROMATIC:
        group = 4
    else:
        group = 1
    channels = np.zeros(15, dtype=np.float64)
    channels[group] = 1.0
    for i, symbol in enumerate(("C", "N", "O", "S", "P", "F", "CL", "BR", "I", "SE")):
        channels[5 + i] = 1.0 if element.upper() == symbol else 0.0
    return channels


def parse_complex(
    protein_path: str | Path,
    ligand_path: str | Path,
    ca_cutoff: float = 10.0,
    cross_cutoff: float = CROSS_CUTOFF,
    base_cutoff: float = 4.5,
) -> Complex:
    """Parse a complex and build the contact graph inputs.

    Protein C-alpha atoms within ``ca_cutoff`` of any ligand heavy atom are
    kept. Edges are born at the protein C-alpha pair distance, at ligand bonds
    (birth zero) or at the protein-ligand contact distance; contacts beyond
    ``cross_cutoff`` are not part of the complex.
    """
    ligand = read_ligand(ligand_path)
    protein_atoms = read_protein_ca(protein_path)
    protein_xyz = np.asarray([atom.xyz for atom in protein_atoms], dtype=np.float64)
    keep = (
        np.min(np.linalg.norm(protein_xyz[:, None, :] - ligand.xyz[None, :, :], axis=2), axis=1)
        <= ca_cutoff
    )
    if not keep.any():
        raise ValueError(f"no protein C-alpha within {ca_cutoff} A of the ligand")
    protein_xyz = protein_xyz[keep]
    protein_atoms = [atom for atom, kept in zip(protein_atoms, keep, strict=True) if kept]
    coordinates = np.vstack([protein_xyz, ligand.xyz])
    n_protein = len(protein_xyz)
    n_ligand = len(ligand.xyz)

    births: dict[tuple[int, int], float] = {}
    cross_edges: list[tuple[int, int]] = []
    cross_coords: list[np.ndarray] = []
    cross_ids: list[tuple[int, int]] = []
    cross_chemistry: list[np.ndarray] = []
    for i, j in combinations(range(n_protein), 2):
        distance = float(np.linalg.norm(protein_xyz[i] - protein_xyz[j]))
        if distance <= base_cutoff:
            births[(i, j)] = 0.0
    for i, j in ligand.bonds:
        births[tuple(sorted((n_protein + i, n_protein + j)))] = 0.0
    for i in range(n_protein):
        for j in range(n_ligand):
            distance = float(np.linalg.norm(protein_xyz[i] - ligand.xyz[j]))
            if distance <= cross_cutoff:
                edge = (i, n_protein + j)
                births[edge] = distance
                cross_edges.append(edge)
                cross_coords.append(np.r_[protein_xyz[i], ligand.xyz[j]])
                cross_ids.append(edge)
                cross_chemistry.append(
                    chemistry_channels(protein_atoms[i].resname, ligand.elements[j])
                )
    if not cross_edges:
        raise ValueError(f"no protein-ligand contact within {cross_cutoff} A")
    return Complex(
        protein_xyz=protein_xyz,
        ligand_xyz=ligand.xyz,
        protein_atoms=protein_atoms,
        ligand=ligand,
        coords=coordinates,
        births=births,
        cross_edges=cross_edges,
        cross_coords=np.asarray(cross_coords, dtype=np.float64),
        cross_ids=np.asarray(cross_ids, dtype=np.int64),
        cross_chemistry=np.asarray(cross_chemistry, dtype=np.float64),
    )


__all__ = [
    "WINDOWS",
    "HEAT_TIMES",
    "BAND_TIMES",
    "Complex",
    "Ligand",
    "ProteinAtom",
    "Window",
    "boundary_matrices",
    "build_window",
    "build_windows",
    "chemistry_channels",
    "flag_triangles",
    "gauss_coupling",
    "metric_from_coupling",
    "parse_complex",
    "persistent_hodge_laplacian",
    "pgl_operator",
    "read_ligand",
    "read_protein_ca",
    "read_protein_heavy",
]
