"""All-atom protein-ligand contact block.

This block is independent of the C-alpha topology branch. It uses protein
heavy atoms and ligand heavy atoms with element, residue-class and
functional-group typing, and summarises the interface as 158 numbers
(121 typed contact counts plus 37 distance and chemistry descriptors).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .operator import read_ligand, read_protein_heavy

ELEMENT_GROUPS = ("C", "N", "O", "S", "P", "F", "CL", "BR", "I", "METAL", "OTHER")
METALS = {
    "ZN",
    "MG",
    "MN",
    "FE",
    "CA",
    "CU",
    "NI",
    "CO",
    "CD",
    "HG",
    "NA",
    "K",
    "CS",
    "BA",
    "SR",
    "YB",
    "GD",
}
DISTANCE_BINS = np.arange(2.0, 7.01, 0.5)  # 10 bins
SHELL_CUTOFF = 8.0
CONTACT_CUTOFF = 4.5

HB_ACCEPTOR_SMARTS = (
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
)
HB_DONOR_SMARTS = "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),$([n;H1;+0])]"
AROMATIC_SMARTS = "a"
POSITIVE_SMARTS = "[+,+2,+3;!$(C(=[O,N,S])-O)]"
NEGATIVE_SMARTS = "[-,-2,-3;!$(C(=[O,N,S])-O)]"

PROTEIN_DONORS = {
    "N",
    "NE",
    "ND1",
    "ND2",
    "NE1",
    "NE2",
    "NH1",
    "NH2",
    "NZ",
    "OG",
    "OG1",
    "OH",
    "SG",
}
PROTEIN_ACCEPTORS = {
    "O",
    "OD1",
    "OD2",
    "OE1",
    "OE2",
    "OG",
    "OG1",
    "OH",
    "ND1",
    "NE2",
    "SD",
    "SG",
}
PROTEIN_AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP"}
PROTEIN_AROMATIC_ATOMS = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HID": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HIE": {"CG", "ND1", "CD2", "CE1", "NE2"},
    "HIP": {"CG", "ND1", "CD2", "CE1", "NE2"},
}
PROTEIN_POSITIVE_RESIDUES = {"LYS", "ARG", "HIS", "HID", "HIE", "HIP"}
PROTEIN_NEGATIVE_RESIDUES = {"ASP", "GLU"}
PROTEIN_HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "ILE", "LEU", "MET", "PRO", "PHE", "TRP", "GLY"}


def _element_group(element: str) -> int:
    element = (element or "X").upper()
    if element in ELEMENT_GROUPS:
        return ELEMENT_GROUPS.index(element)
    if element in METALS:
        return ELEMENT_GROUPS.index("METAL")
    return ELEMENT_GROUPS.index("OTHER")


def _ligand_flags(mol):
    """Return ligand donor, acceptor, aromatic, positive and negative atom sets."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    empty = (set(), set(), set(), set(), set())
    try:
        if mol.HasProp("pgl_ligand_sanitized") and mol.GetProp("pgl_ligand_sanitized") == "0":
            return empty
    except Exception:  # noqa: BLE001
        return empty

    def matches(smarts: str) -> set[int]:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                return set()
            found: set[int] = set()
            for match in mol.GetSubstructMatches(pattern, useChirality=False):
                found.update(match)
            return found
        except Exception:  # noqa: BLE001
            return set()

    try:
        donors = matches(HB_DONOR_SMARTS)
        acceptors = matches(HB_ACCEPTOR_SMARTS)
        aromatic = matches(AROMATIC_SMARTS)
        positive = matches(POSITIVE_SMARTS)
        negative = matches(NEGATIVE_SMARTS)
    except Exception:  # noqa: BLE001
        donors = acceptors = aromatic = positive = negative = set()
    return donors, acceptors, aromatic, positive, negative


def contact_features(protein_path: str | Path, ligand_path: str | Path) -> np.ndarray:
    """Compute the 158-dimensional all-atom contact block.

    The protein side uses ATOM records of standard residues within 8 A of the
    ligand; contacts are counted within 4.5 A. Functional-group flags that
    cannot be derived for an unsanitized ligand are left at zero.
    """
    ligand = read_ligand(ligand_path)
    xyz = ligand.xyz
    protein = read_protein_heavy(protein_path)
    if not protein:
        raise ValueError(f"no protein heavy atoms in {protein_path}")
    protein_xyz = np.asarray([atom.xyz for atom in protein], dtype=np.float64)
    n_ligand = len(xyz)
    donors, acceptors, aromatic, positive, negative = _ligand_flags(ligand.mol)
    heavy_index = [atom.GetIdx() for atom in ligand.mol.GetAtoms() if atom.GetAtomicNum() > 1]
    renumber = {old: new for new, old in enumerate(heavy_index)}
    donors = {renumber[i] for i in donors if i in renumber}
    acceptors = {renumber[i] for i in acceptors if i in renumber}
    aromatic = {renumber[i] for i in aromatic if i in renumber}
    positive = {renumber[i] for i in positive if i in renumber}
    negative = {renumber[i] for i in negative if i in renumber}

    distances = np.linalg.norm(protein_xyz[:, None, :] - xyz[None, :, :], axis=2)
    shell = distances.min(axis=1) <= SHELL_CUTOFF
    shell_atoms = [atom for atom, kept in zip(protein, shell, strict=True) if kept]
    shell_distances = distances[shell]

    ligand_group = np.asarray([_element_group(e) for e in ligand.elements])
    protein_group = np.asarray([_element_group(atom.element) for atom in shell_atoms])
    within = shell_distances <= CONTACT_CUTOFF
    counts = np.zeros((len(ELEMENT_GROUPS), len(ELEMENT_GROUPS)), dtype=np.float64)
    if within.any():
        protein_index, ligand_index = np.nonzero(within)
        np.add.at(counts, (protein_group[protein_index], ligand_group[ligand_index]), 1.0)
    counts = counts.reshape(-1) / max(n_ligand, 1)

    ligand_min = shell_distances.min(axis=0)
    protein_min = shell_distances.min(axis=1)
    ligand_hist = np.histogram(ligand_min, bins=DISTANCE_BINS)[0].astype(np.float64) / max(
        n_ligand, 1
    )
    protein_hist = np.histogram(protein_min, bins=DISTANCE_BINS)[0].astype(np.float64) / max(
        len(protein_min), 1
    )
    distance_stats = np.asarray(
        [ligand_min.min(), np.quantile(ligand_min, 0.05), np.median(ligand_min), ligand_min.mean()]
    )

    protein_donor = np.asarray([atom.name in PROTEIN_DONORS for atom in shell_atoms])
    protein_acceptor = np.asarray([atom.name in PROTEIN_ACCEPTORS for atom in shell_atoms])
    ligand_donor = np.zeros(n_ligand, dtype=bool)
    ligand_acceptor = np.zeros(n_ligand, dtype=bool)
    for index in donors:
        ligand_donor[index] = True
    for index in acceptors:
        ligand_acceptor[index] = True
    hb_protein_donor = (
        int(np.sum(within[np.ix_(protein_donor, ligand_acceptor)]))
        if protein_donor.any() and ligand_acceptor.any()
        else 0
    )
    hb_ligand_donor = (
        int(np.sum(within[np.ix_(protein_acceptor, ligand_donor)]))
        if protein_acceptor.any() and ligand_donor.any()
        else 0
    )
    hb = np.asarray([hb_protein_donor, hb_ligand_donor, hb_protein_donor + hb_ligand_donor])

    protein_aromatic = np.asarray(
        [
            atom.resname in PROTEIN_AROMATIC_RESIDUES
            and atom.name in PROTEIN_AROMATIC_ATOMS.get(atom.resname, set())
            for atom in shell_atoms
        ]
    )
    ligand_aromatic = np.zeros(n_ligand, dtype=bool)
    for index in aromatic:
        ligand_aromatic[index] = True
    aromatic_count = (
        int(np.sum(within[np.ix_(protein_aromatic, ligand_aromatic)]))
        if protein_aromatic.any() and ligand_aromatic.any()
        else 0
    )
    aromatic_frac = aromatic_count / max(n_ligand, 1)

    protein_positive = np.asarray(
        [atom.resname in PROTEIN_POSITIVE_RESIDUES for atom in shell_atoms]
    )
    protein_negative = np.asarray(
        [atom.resname in PROTEIN_NEGATIVE_RESIDUES for atom in shell_atoms]
    )
    ligand_positive = np.zeros(n_ligand, dtype=bool)
    ligand_negative = np.zeros(n_ligand, dtype=bool)
    for index in positive:
        ligand_positive[index] = True
    for index in negative:
        ligand_negative[index] = True
    charged = []
    charged.append(
        int(np.sum(within[np.ix_(protein_positive, ligand_negative)]))
        if protein_positive.any() and ligand_negative.any()
        else 0
    )
    charged.append(
        int(np.sum(within[np.ix_(protein_negative, ligand_positive)]))
        if protein_negative.any() and ligand_positive.any()
        else 0
    )
    same_charge = 0
    if protein_positive.any() and ligand_positive.any():
        same_charge += int(np.sum(within[np.ix_(protein_positive, ligand_positive)]))
    if protein_negative.any() and ligand_negative.any():
        same_charge += int(np.sum(within[np.ix_(protein_negative, ligand_negative)]))
    charged.append(same_charge)
    charged.append(sum(charged[:3]) / max(n_ligand, 1))

    protein_hydrophobic = np.asarray(
        [
            atom.resname in PROTEIN_HYDROPHOBIC_RESIDUES and atom.element == "C"
            for atom in shell_atoms
        ]
    )
    ligand_carbon = np.asarray([element == "C" for element in ligand.elements])
    hydrophobic_count = (
        int(np.sum(within[np.ix_(protein_hydrophobic, ligand_carbon)]))
        if protein_hydrophobic.any() and ligand_carbon.any()
        else 0
    )

    density = float(within.sum()) / max(n_ligand * max(len(shell_atoms), 1), 1)
    tail = np.asarray([density, len(shell_atoms), n_ligand], dtype=np.float64)
    block = np.concatenate(
        [
            counts,
            ligand_hist,
            protein_hist,
            distance_stats,
            hb,
            np.asarray([aromatic_count, aromatic_frac]),
            np.asarray(charged, dtype=np.float64),
            np.asarray([hydrophobic_count], dtype=np.float64),
            tail,
        ]
    )
    if block.shape != (158,):
        raise RuntimeError(f"contact block has width {block.shape}, expected (158,)")
    return block


def feature_names() -> list[str]:
    """Names of the 158 contact features, in block order."""
    names = [f"contact_{a}_{b}_4p5" for a in ELEMENT_GROUPS for b in ELEMENT_GROUPS]
    names += [f"lig_min_dist_hist_{i}" for i in range(10)]
    names += [f"prot_min_dist_hist_{i}" for i in range(10)]
    names += ["d_min", "d_q05", "d_median", "d_mean"]
    names += ["hb_prot_don_lig_acc_3p5", "hb_lig_don_prot_acc_3p5", "hb_like_total_3p5"]
    names += ["arom_contact_4p5", "arom_contact_frac"]
    names += ["charged_pos_neg_4p5", "charged_neg_pos_4p5", "charged_same_4p5", "charged_frac"]
    names += ["hydrophobic_c_c_4p0"]
    names += ["contact_density", "n_protein_atoms_shell", "n_ligand_heavy"]
    return names


__all__ = ["ELEMENT_GROUPS", "contact_features", "feature_names"]
