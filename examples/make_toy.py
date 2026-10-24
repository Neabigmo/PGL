"""Generate the toy dataset used by the quickstart.

The complexes are synthetic: an idealised polypeptide helix and a small
ligand placed next to its surface. They exercise every block of the
representation (contact topology, Gauss coupling, all-atom contacts) but carry
no scientific information; the affinity labels are a fixed function of the
computed features so that the readout has something to fit.

Run from the repository root after installing the package:

    python examples/make_toy.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy"
COMPLEXES = TOY / "complexes"

# Residues cycle through the chemistry classes used by the first five channels.
RESIDUES = ("ALA", "LEU", "SER", "LYS", "ASP", "PHE", "GLY", "THR", "ARG", "TYR")
LIGANDS = (
    "c1ccccc1",
    "c1ccccc1O",
    "c1ccncc1",
    "CC(=O)Nc1ccc(O)cc1",
    "OC(=O)c1ccccc1",
    "c1ccc2[nH]ccc2c1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Nc1ncnc2[nH]cnc12",
    "c1ccc(cc1)Cc1ccccc1",
    "CS(=O)(=O)Nc1ccc(O)cc1",
    "c1cc2ccccc2[nH]1",
    "CC(=O)Oc1ccccc1C(=O)O",
)
# (residues, ligand distance from the helix axis, residue phase shift)
VARIANTS = ((18, 5.0, 0), (21, 5.6, 2), (24, 4.6, 4), (27, 6.1, 6), (30, 5.2, 8))
RADIUS = 2.3
RISE = 1.5


def _helix_frame(index: int):
    theta = math.radians(100.0 * index)
    position = np.array([RADIUS * math.cos(theta), RADIUS * math.sin(theta), RISE * index])
    radial = np.array([math.cos(theta), math.sin(theta), 0.0])
    tangent = np.array(
        [
            -RADIUS * math.sin(theta) * math.radians(100.0),
            RADIUS * math.cos(theta) * math.radians(100.0),
            RISE,
        ]
    )
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.cross(tangent, radial)
    normal = normal / np.linalg.norm(normal)
    return position, radial, tangent, normal


def _protein_atoms(n_residues: int, phase: int):
    """Return (resname, atom_name, element, xyz) for an idealised helix backbone."""
    atoms = []
    for index in range(n_residues):
        position, radial, tangent, normal = _helix_frame(index)
        resname = RESIDUES[(index + phase) % len(RESIDUES)]
        atoms.append((resname, "N", "N", position - 1.0 * tangent + 0.4 * normal))
        atoms.append((resname, "CA", "C", position))
        atoms.append((resname, "C", "C", position + 1.0 * tangent + 0.4 * normal))
        atoms.append((resname, "O", "O", position + 1.0 * tangent + 1.4 * normal))
        atoms.append((resname, "CB", "C", position + 1.5 * radial - 0.3 * tangent))
    return atoms


def _ligand(smiles: str, target: np.ndarray, seed: int):
    """Embed a ligand and move its centroid to ``target``."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"cannot parse {smiles}")
    mol = Chem.AddHs(mol)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    if AllChem.EmbedMolecule(mol, parameters) != 0:
        raise RuntimeError(f"embedding failed for {smiles}")
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
    coordinates -= coordinates.mean(axis=0)
    coordinates += target
    for index in range(mol.GetNumAtoms()):
        conformer.SetAtomPosition(index, coordinates[index])
    return Chem.RemoveHs(mol)


def _write_pdb(path: Path, atoms) -> None:
    serial = 0
    residue_index = 0
    lines = []
    for resname, name, element, xyz in atoms:
        if name == "N":
            residue_index += 1
        serial += 1
        lines.append(
            f"ATOM  {serial:5d} {name:<4s} {resname:>3s} A{residue_index:4d}"
            f"    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            f"{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sdf(path: Path, mol) -> None:
    path.write_text(Chem.MolToMolBlock(mol), encoding="utf-8")


def _label(tokens: np.ndarray, index: int) -> float:
    """A fixed, feature-driven label for the demo readout."""
    aromatic = float(tokens[4])
    ligand_carbon = float(tokens[5])
    d_min = float(tokens[270])
    return 4.5 + 1.5 * aromatic - 0.5 * (d_min - 3.5) + 0.3 * ligand_carbon + 0.03 * index


def main() -> int:
    from pgl.features import contact_features

    if TOY.exists():
        for path in sorted(TOY.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
    COMPLEXES.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    labels_rows = []
    index = 0
    for n_residues, distance, phase in VARIANTS:
        atoms = _protein_atoms(n_residues, phase)
        for ligand_index, smiles in enumerate(LIGANDS):
            sample_id = f"toy{index + 1:02d}"
            protein_path = COMPLEXES / f"{sample_id}_protein.pdb"
            ligand_path = COMPLEXES / f"{sample_id}_ligand.sdf"
            anchor = max(2, min(n_residues - 3, n_residues // 3 + ligand_index % 6))
            position, radial, _tangent, _normal = _helix_frame(anchor)
            _write_pdb(protein_path, atoms)
            # Pull the ligand towards the helix until it has contacts in the
            # central filtration window, where the contact tokens are defined.
            tokens = None
            for attempt in range(4):
                target = position + (distance - 0.4 * attempt) * radial
                mol = _ligand(smiles, target, seed=1000 + index + 17 * attempt)
                _write_sdf(ligand_path, mol)
                try:
                    tokens = contact_features(protein_path, ligand_path).vector
                except ValueError:
                    continue
                break
            if tokens is None:
                raise RuntimeError(f"{sample_id}: no contact after placement retries")
            label = _label(tokens, index)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "protein_path": protein_path.relative_to(ROOT).as_posix(),
                    "ligand_path": ligand_path.relative_to(ROOT).as_posix(),
                }
            )
            labels_rows.append({"sample_id": sample_id, "label": f"{label:.4f}"})
            index += 1

    with open(TOY / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "protein_path", "ligand_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    with open(TOY / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label"])
        writer.writeheader()
        writer.writerows(labels_rows)
    print(f"wrote {len(manifest_rows)} toy complexes to {TOY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
