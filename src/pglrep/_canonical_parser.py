"""Profile the fixed PGL-Rep GPU v1 numerical path on real PDBbind complexes.

This is a benchmark harness only.  It imports the existing PGL-Rep core and
does not alter its mathematical definitions or the fixed 1137-D readout.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
import torch

from ._canonical_gpu_core import (
    PGLConfig,
    _spectral_signal_gram_base,
    embed_cross_gauss_into_all_edges,
    gauss_linking_matrix_gpu,
    gauss_metric_dense,
    global_spectral_features,
    local_gauss_context,
    local_hks_wavelet_from_eigs,
    pointcloud32,
    pgl_symmetric_dense_from_persistent_up,
    soft_assign,
    spectral_signal_response,
    token_mean_std,
    up_laplacian_from_triangles_gpu,
    vlad_pool,
)
from ._canonical_pipeline import ScaleInput, Vocabulary


STAGES = [
    "data_topology",
    "gauss_edge_pair_integral",
    "WGLGM_construction",
    "persistent_up_schur",
    "MG_decomp_PGL_eig",
    "global_spectral_features",
    "HKS_wavelet",
    "chemical_spectral_response",
    "point_cloud_token",
    "VLAD_context",
]


@dataclass
class RawScale:
    B1: np.ndarray
    tri_rows: np.ndarray
    tri_signs: np.ndarray
    n_edges_y: int
    beta1: int
    cross_rows: np.ndarray
    p_start: np.ndarray
    l_end: np.ndarray
    endpoint_ids: np.ndarray
    chem15: np.ndarray


@dataclass
class DeviceScale:
    scale: ScaleInput
    tri_rows: torch.Tensor
    tri_signs: torch.Tensor
    n_edges_y: int


def _parse_pdb(path: Path) -> Tuple[np.ndarray, List[str]]:
    coords: List[List[float]] = []
    residues: List[str] = []
    seen = set()
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        alt = line[16:17].strip()
        if alt not in ("", "A", "1"):
            continue
        key = (line[21:22], line[22:26].strip(), line[26:27])
        if key in seen:
            continue
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        seen.add(key)
        coords.append(xyz)
        residues.append(line[17:20].strip().upper())
    if not coords:
        raise ValueError(f"No protein CA atoms found in {path}")
    return np.asarray(coords, dtype=np.float32), residues


def _parse_sdf(path: Path) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]]]:
    lines = path.read_text(errors="replace").splitlines()
    counts = next((i for i, line in enumerate(lines[:12]) if "V2000" in line or "V3000" in line), None)
    if counts is None:
        raise ValueError(f"No SDF counts line found in {path}")
    if "V3000" in lines[counts]:
        raise ValueError(f"V3000 SDF is not supported by the lightweight parser: {path}")
    try:
        n_atoms = int(lines[counts][:3])
        n_bonds = int(lines[counts][3:6])
    except ValueError as exc:
        raise ValueError(f"Bad SDF counts line in {path}") from exc
    atom_lines = lines[counts + 1:counts + 1 + n_atoms]
    bond_lines = lines[counts + 1 + n_atoms:counts + 1 + n_atoms + n_bonds]
    xyz: List[List[float]] = []
    elements: List[str] = []
    for line in atom_lines:
        try:
            xyz.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
            elements.append(line[31:34].strip().upper())
        except ValueError:
            continue
    if len(xyz) != n_atoms:
        raise ValueError(f"Could not parse all SDF atoms in {path}")
    bonds: List[Tuple[int, int]] = []
    for line in bond_lines:
        try:
            a, b = int(line[:3]) - 1, int(line[3:6]) - 1
        except ValueError:
            continue
        bonds.append((a, b))
    # Canonicalize the ligand atom record order before any geometry/topology
    # construction.  The representation must not depend on SDF record order.
    # Use graph and pairwise-distance signatures; unlike an arbitrary
    # canonical-rank tie break, these are invariant to record permutation and
    # to a rigid rotation of the same coordinates.
    try:
        from rdkit import Chem

        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
        ranks = (
            np.asarray(list(Chem.CanonicalRankAtoms(mol, breakTies=False)), dtype=np.int64)
            if mol is not None and mol.GetNumAtoms() == n_atoms
            else np.zeros(n_atoms, dtype=np.int64)
        )
        xyz_arr = np.asarray(xyz, dtype=np.float32)
        distance_matrix = np.linalg.norm(
            xyz_arr[:, None, :] - xyz_arr[None, :, :], axis=2
        )
        adjacency = [[] for _ in range(n_atoms)]
        for a, b in bonds:
            adjacency[a].append(b)
            adjacency[b].append(a)

        def atom_key(i: int):
            neighbour_elements = tuple(sorted(elements[j] for j in adjacency[i]))
            distance_signature = tuple(np.round(np.sort(distance_matrix[i]), 3).tolist())
            return (
                elements[i],
                len(adjacency[i]),
                neighbour_elements,
                distance_signature,
                int(ranks[i]),
            )

        order = np.asarray(sorted(range(n_atoms), key=atom_key), dtype=np.int64)
        if len(order) == n_atoms:
            old_to_new = {int(old): int(new) for new, old in enumerate(order)}
            xyz_arr = xyz_arr[order]
            elements = [elements[int(old)] for old in order]
            bonds = [(old_to_new[a], old_to_new[b]) for a, b in bonds]
            return xyz_arr, elements, bonds
    except Exception:
        # Keep the lightweight parser usable for environments without RDKit;
        # the audit records that fallback through the resulting invariance
        # report rather than silently changing the data.
        pass
    return np.asarray(xyz, dtype=np.float32), elements, bonds


def _residue_class(res: str) -> int:
    if res in {"ALA", "VAL", "ILE", "LEU", "MET", "PRO", "GLY"}:
        return 0
    if res in {"SER", "THR", "ASN", "GLN", "TYR"}:
        return 1
    if res in {"LYS", "ARG", "HIS"}:
        return 2
    if res in {"ASP", "GLU"}:
        return 3
    return 4


def _element_class(element: str) -> int:
    return {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4}.get(element, 5)


def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _distance_order(values: np.ndarray, indices: Sequence[int]) -> List[int]:
    """Stable geometry ordering robust to text-format rounding."""
    return sorted(indices, key=lambda i: (round(float(values[i]), 3), int(i)))


def _boundary_matrix(n_vertices: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    B = np.zeros((n_vertices, len(edges)), dtype=np.float32)
    for j, (a, b) in enumerate(edges):
        if a > b:
            a, b = b, a
        B[a, j] = -1.0
        B[b, j] = 1.0
    return B


def _triangles_for_graph(
    n_vertices: int,
    edges: Sequence[Tuple[int, int]],
    max_triangles: int = 192,
    neighbor_cap: int = 18,
) -> Tuple[np.ndarray, np.ndarray]:
    edge_to_row = {_edge_key(a, b): i for i, (a, b) in enumerate(edges)}
    adj = [set() for _ in range(n_vertices)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    rows: List[List[int]] = []
    signs: List[List[float]] = []
    seen = set()
    for a in range(n_vertices):
        neigh = sorted(adj[a])[:neighbor_cap]
        for ii, b in enumerate(neigh):
            for c in neigh[ii + 1:]:
                tri = tuple(sorted((a, b, c)))
                if tri in seen or _edge_key(b, c) not in edge_to_row:
                    continue
                seen.add(tri)
                x, y, z = tri
                # Boundary [y,z] - [x,z] + [x,y], all graph edges are low->high.
                rows.append([edge_to_row[(y, z)], edge_to_row[(x, z)], edge_to_row[(x, y)]])
                signs.append([1.0, -1.0, 1.0])
                if len(rows) >= max_triangles:
                    return np.asarray(rows, dtype=np.int64), np.asarray(signs, dtype=np.float32)
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3), np.asarray(signs, dtype=np.float32).reshape(-1, 3)


def _persistent_beta1(B1: np.ndarray, x_edges: Sequence[Tuple[int, int]], y_edges: Sequence[Tuple[int, int]], tri_rows: np.ndarray, tri_signs: np.ndarray) -> int:
    if len(tri_rows) == 0:
        return int(B1.shape[1] - np.linalg.matrix_rank(B1.astype(np.float64)))
    B2Y = np.zeros((len(y_edges), len(tri_rows)), dtype=np.float64)
    for t in range(len(tri_rows)):
        B2Y[tri_rows[t], t] = tri_signs[t]
    BX = B2Y[:len(x_edges)]
    Bout = B2Y[len(x_edges):]
    Z = np.eye(B2Y.shape[1], dtype=np.float64) if Bout.shape[0] == 0 else scipy.linalg.null_space(Bout)
    B2p = BX @ Z
    return int(B1.shape[1] - np.linalg.matrix_rank(B1.astype(np.float64)) - np.linalg.matrix_rank(B2p))


def _make_chem15(
    distances: np.ndarray,
    residues: Sequence[str],
    elements: Sequence[str],
    p_indices: Sequence[int],
    l_indices: Sequence[int],
) -> np.ndarray:
    out = np.zeros((len(distances), 15), dtype=np.float32)
    for i, (d, p, l) in enumerate(zip(distances, p_indices, l_indices)):
        out[i, _element_class(elements[l])] = 1.0
        out[i, 6 + _residue_class(residues[p])] = 1.0
        dbin = 0 if d < 3.5 else 1 if d < 5.0 else 2 if d < 6.5 else 3
        out[i, 11 + dbin] = 1.0
    return out


def prepare_raw_complex(path: Path, max_protein: int = 96, max_ligand: int = 48, max_cross: int = 96, coordinate_override: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[List[RawScale], np.ndarray, np.ndarray, Dict[str, int]]:
    # Accept both the original PDBbind names (e.g. ``1abc_protein.pdb``)
    # and the frozen-project names (``protein.pdb`` / ``ligand.sdf``).
    pdb_candidates = list(path.glob("*_protein.pdb")) or list(path.glob("protein.pdb"))
    sdf_candidates = list(path.glob("*_ligand.sdf")) or list(path.glob("ligand.sdf"))
    if not pdb_candidates or not sdf_candidates:
        raise FileNotFoundError(f"expected protein PDB and ligand SDF in {path}")
    pdb = pdb_candidates[0]
    sdf = sdf_candidates[0]
    p_all, residues_all = _parse_pdb(pdb)
    l_all, elements_all, ligand_bonds = _parse_sdf(sdf)
    if coordinate_override is not None:
        p_override, l_override = coordinate_override
        if p_override.shape != p_all.shape or l_override.shape != l_all.shape:
            raise ValueError(f"in-memory coordinate shape mismatch: {p_override.shape}/{l_override.shape} != {p_all.shape}/{l_all.shape}")
        p_all = np.asarray(p_override, dtype=float)
        l_all = np.asarray(l_override, dtype=float)
    l_keep = [i for i, e in enumerate(elements_all) if e != "H"]
    if len(l_keep) > max_ligand:
        l_center = l_all[l_keep].mean(axis=0)
        l_dist = np.linalg.norm(l_all - l_center[None, :], axis=1)
        l_keep = _distance_order(l_dist, l_keep)[:max_ligand]
    l_xyz = l_all[l_keep]
    l_elements = [elements_all[i] for i in l_keep]
    l_remap = {old: new for new, old in enumerate(l_keep)}
    p_center = l_xyz.mean(axis=0)
    p_dist = np.linalg.norm(p_all - p_center[None, :], axis=1)
    p_keep = np.where(p_dist <= 16.0)[0].tolist()
    if len(p_keep) < 8:
        p_keep = _distance_order(p_dist, range(len(p_all)))[:min(max_protein, len(p_all))]
    if len(p_keep) > max_protein:
        p_keep = _distance_order(p_dist, p_keep)[:max_protein]
    p_keep = sorted(p_keep)
    p_xyz = p_all[p_keep]
    p_residues = [residues_all[i] for i in p_keep]
    p_remap = {old: new for new, old in enumerate(p_keep)}
    n_p = len(p_xyz)
    n_l = len(l_xyz)
    n_v = n_p + n_l

    base = set()
    # Sequential CA edges in the pocket.
    for a, b in zip(range(n_p - 1), range(1, n_p)):
        if p_keep[b] - p_keep[a] <= 3:
            base.add(_edge_key(a, b))
    # A small real-coordinate kNN graph adds local protein topology.
    pd2 = ((p_xyz[:, None, :] - p_xyz[None, :, :]) ** 2).sum(axis=2)
    for a in range(n_p):
        for b in np.argsort(pd2[a], kind="stable")[1:5]:
            if pd2[a, b] <= 10.0 ** 2:
                base.add(_edge_key(a, int(b)))
    # Ligand bonds from the SDF, restricted to retained heavy atoms.
    for old_a, old_b in ligand_bonds:
        if old_a in l_remap and old_b in l_remap:
            base.add(_edge_key(n_p + l_remap[old_a], n_p + l_remap[old_b]))
    if not any(a >= n_p and b >= n_p for a, b in base) and n_l > 1:
        ld2 = ((l_xyz[:, None, :] - l_xyz[None, :, :]) ** 2).sum(axis=2)
        for a in range(n_l):
            b = int(np.argsort(ld2[a])[1])
            base.add(_edge_key(n_p + a, n_p + b))

    cd = np.linalg.norm(p_xyz[:, None, :] - l_xyz[None, :, :], axis=2)
    candidates = [(float(cd[p, l]), p, l) for p in range(n_p) for l in range(n_l) if cd[p, l] <= 7.0]
    if not candidates:
        candidates = [(float(cd[p, l]), p, l) for p in range(n_p) for l in range(n_l)]
    candidates.sort(key=lambda x: (round(x[0], 3), x[1], x[2]))
    candidates = candidates[:max_cross]
    cross_dist = np.asarray([x[0] for x in candidates], dtype=np.float32)
    cross_p = [x[1] for x in candidates]
    cross_l = [x[2] for x in candidates]
    chem_all = _make_chem15(cross_dist, p_residues, l_elements, cross_p, cross_l)
    thresholds = (4.0, 5.0, 6.0, 7.0)
    raw_scales: List[RawScale] = []
    for xthr, ythr in zip(thresholds[:3], thresholds[1:]):
        x_ids = [i for i, d in enumerate(cross_dist) if d <= xthr]
        y_ids = [i for i, d in enumerate(cross_dist) if d <= ythr]
        if len(x_ids) < 4:
            x_ids = sorted(set(x_ids + list(range(min(4, len(candidates))))) )
        if not y_ids:
            y_ids = x_ids[:]
        x_edges = sorted(base) + [_edge_key(p, n_p + l) for p, l in zip([cross_p[i] for i in x_ids], [cross_l[i] for i in x_ids])]
        y_extra = [_edge_key(cross_p[i], n_p + cross_l[i]) for i in y_ids if _edge_key(cross_p[i], n_p + cross_l[i]) not in set(x_edges)]
        y_edges = x_edges + sorted(set(y_extra))
        x_map = {e: i for i, e in enumerate(x_edges)}
        y_map = {e: i for i, e in enumerate(y_edges)}
        y_tri_local, tri_signs = _triangles_for_graph(n_v, y_edges)
        tri_rows = y_tri_local
        B1 = _boundary_matrix(n_v, x_edges)
        cross_edges = [_edge_key(cross_p[i], n_p + cross_l[i]) for i in x_ids]
        cross_rows = np.asarray([x_map[e] for e in cross_edges], dtype=np.int64)
        p_start = p_xyz[np.asarray(cross_p)[x_ids]]
        l_end = l_xyz[np.asarray(cross_l)[x_ids]]
        endpoint_ids = np.asarray([[cross_p[i], n_p + cross_l[i]] for i in x_ids], dtype=np.int64)
        chem = chem_all[x_ids]
        beta1 = _persistent_beta1(B1, x_edges, y_edges, tri_rows, tri_signs)
        raw_scales.append(RawScale(B1, tri_rows, tri_signs, len(y_edges), beta1, cross_rows, p_start, l_end, endpoint_ids, chem))
    meta = {
        "protein_ca": n_p,
        "ligand_heavy": n_l,
        "edge_s1": raw_scales[0].B1.shape[1],
        "edge_s2": raw_scales[1].B1.shape[1],
        "edge_s3": raw_scales[2].B1.shape[1],
        "tri_s1": raw_scales[0].tri_rows.shape[0],
        "tri_s2": raw_scales[1].tri_rows.shape[0],
        "tri_s3": raw_scales[2].tri_rows.shape[0],
        "dim_s1": raw_scales[0].B1.shape[1],
        "dim_s2": raw_scales[1].B1.shape[1],
        "dim_s3": raw_scales[2].B1.shape[1],
    }
    return raw_scales, p_xyz, l_xyz, meta


def to_device_inputs(raw: Sequence[RawScale], p_xyz: np.ndarray, l_xyz: np.ndarray, device: torch.device, cfg: PGLConfig) -> Tuple[List[DeviceScale], torch.Tensor, torch.Tensor]:
    out: List[DeviceScale] = []
    for s in raw:
        b1 = torch.as_tensor(s.B1, device=device, dtype=cfg.dtype)
        empty_b2 = torch.empty((s.B1.shape[1], 0), device=device, dtype=cfg.dtype)
        si = ScaleInput(
            B1=b1, B2p=empty_b2, beta1=s.beta1,
            cross_rows_x=torch.as_tensor(s.cross_rows, device=device, dtype=torch.long),
            p_start=torch.as_tensor(s.p_start, device=device, dtype=cfg.dtype),
            l_end=torch.as_tensor(s.l_end, device=device, dtype=cfg.dtype),
            endpoint_ids=torch.as_tensor(s.endpoint_ids, device=device, dtype=torch.long),
            chem15_cross=torch.as_tensor(s.chem15, device=device, dtype=cfg.dtype),
        )
        out.append(DeviceScale(si, torch.as_tensor(s.tri_rows, device=device, dtype=torch.long), torch.as_tensor(s.tri_signs, device=device, dtype=cfg.dtype), s.n_edges_y))
    return out, torch.as_tensor(p_xyz, device=device, dtype=cfg.dtype), torch.as_tensor(l_xyz, device=device, dtype=cfg.dtype)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, fn):
    _sync(device)
    t0 = time.perf_counter()
    value = fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0, value


def run_once(
    path: Path,
    device: torch.device,
    cfg: PGLConfig,
    vocab: Vocabulary,
    return_features: bool = False,
    ablation: str = "full",
    token_keep: Optional[Sequence[int]] = None,
    coordinate_override: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, object]:
    valid_ablations = {"full", "pl", "non_persistent", "no_chem_metric", "gauss_shuffle", "gauss_shuffle_25", "gauss_shuffle_50", "gauss_shuffle_75"}
    if ablation not in valid_ablations:
        raise ValueError(f"unknown ablation {ablation}; expected one of {sorted(valid_ablations)}")
    drop_gauss_local = ablation == "pl"
    shuffle_fraction = {"gauss_shuffle": 1.0, "gauss_shuffle_25": 0.25, "gauss_shuffle_50": 0.50, "gauss_shuffle_75": 0.75}.get(ablation, 0.0)
    stage_ms: Dict[str, float] = {}

    def prep():
        raw, p_xyz, l_xyz, meta = prepare_raw_complex(path, coordinate_override=coordinate_override)
        ds, p_dev, l_dev = to_device_inputs(raw, p_xyz, l_xyz, device, cfg)
        return ds, p_dev, l_dev, meta

    stage_ms[STAGES[0]], (scales, protein_xyz, ligand_xyz, meta) = _timed(device, prep)
    gauss = []
    def gauss_():
        result = []
        for s in scales:
            chem_q = torch.ones_like(s.scale.chem15_cross) if ablation == "no_chem_metric" else s.scale.chem15_cross
            wc, sc = gauss_linking_matrix_gpu(s.scale.p_start, s.scale.l_end, s.scale.endpoint_ids, chem_q, cfg, return_signed=True)
            if shuffle_fraction and wc.numel():
                n = wc.shape[0]
                k = max(2, min(n, int(round(n * shuffle_fraction))))
                gen = torch.Generator(device=device).manual_seed(20260824 + n)
                chosen = torch.randperm(n, device=device, generator=gen)[:k]
                local_perm = torch.randperm(k, device=device, generator=torch.Generator(device=device).manual_seed(20260825 + n))
                perm = torch.arange(n, device=device)
                perm[chosen] = chosen[local_perm]
                wc, sc = wc[perm][:, perm], sc[perm][:, perm]
            result.append((wc, sc))
        return result
    stage_ms[STAGES[1]], gauss = _timed(device, gauss_)
    all_blocks = []

    def construct():
        result = []
        for s, (wc, sc) in zip(scales, gauss):
            wall = embed_cross_gauss_into_all_edges(s.scale.n_edges_x, s.scale.cross_rows_x, wc)
            fall = torch.zeros((s.scale.n_edges_x, 15), device=device, dtype=cfg.dtype)
            fall[s.scale.cross_rows_x] = s.scale.chem15_cross
            metric_lambda = 0.0 if ablation == "pl" else cfg.gauss_lambda
            m, lg, gd = gauss_metric_dense(wall, metric_lambda)
            result.append((wall, fall, m, lg, gd))
        return result

    stage_ms[STAGES[2]], constructed = _timed(device, construct)

    def persistent():
        result = []
        for s in scales:
            if ablation == "non_persistent":
                # Ordinary Delta(K5): retain only triangles whose three edges
                # already belong to the lower filtration X=K5.
                keep = torch.all(s.tri_rows < s.scale.n_edges_x, dim=1)
                hup = up_laplacian_from_triangles_gpu(
                    s.tri_rows[keep], s.tri_signs[keep], s.scale.n_edges_x, cfg.dtype
                )
            else:
                lup = up_laplacian_from_triangles_gpu(s.tri_rows, s.tri_signs, s.n_edges_y, cfg.dtype)
                hup = persistent_up_schur_dense_gpu(lup, s.scale.n_edges_x)
            result.append(hup)
        return result

    # The imported function is intentionally resolved here to keep the exact core implementation visible.
    from ._canonical_gpu_core import persistent_up_schur_dense_gpu
    stage_ms[STAGES[3]], hups = _timed(device, persistent)

    def eigs():
        result = []
        for s, (_, _, m, _, _), hup in zip(scales, constructed, hups):
            spectral_dtype = cfg.spectral_dtype if cfg.spectral_mode == "fp64_stable" else cfg.dtype
            result.append(pgl_symmetric_dense_from_persistent_up(s.scale.B1, hup, m, spectral_dtype=spectral_dtype))
        return result

    stage_ms[STAGES[4]], eig_results = _timed(device, eigs)

    def globals_():
        return [global_spectral_features(ev, scale, s.scale.beta1, cfg) for s, (_, ev, _, scale) in zip(scales, eig_results)]
    stage_ms[STAGES[5]], global_blocks = _timed(device, globals_)

    def hks_():
        return [local_hks_wavelet_from_eigs(ev, u, cfg) for _, ev, u, _ in eig_results]
    stage_ms[STAGES[6]], local_blocks = _timed(device, hks_)

    def chem_():
        return [spectral_signal_response(fall, ev, u, cfg.response_times, modes=cfg.response_modes) for (_, fall, _, _, _), (_, ev, u, _) in zip(constructed, eig_results)]
    stage_ms[STAGES[7]], chem_blocks = _timed(device, chem_)

    central = scales[1]
    wc, sc = gauss[1]
    def tokens_():
        pc32 = pointcloud32(central.scale.p_start, central.scale.l_end, protein_xyz, ligand_xyz)
        g3 = local_gauss_context(wc, sc)
        token_parts = [central.scale.chem15_cross, local_blocks[1][central.scale.cross_rows_x], pc32]
        if not drop_gauss_local:
            token_parts.append(g3)
        tok = torch.cat(token_parts, dim=1)
        if token_keep is not None:
            keep = torch.as_tensor(token_keep, device=device, dtype=torch.long)
            tok = tok.index_select(1, keep)
        tok_stats = token_mean_std(tok)
        return pc32, g3, tok, tok_stats
    stage_ms[STAGES[8]], (pc32, g3, tok57, tok114) = _timed(device, tokens_)

    def vlad_context():
        z = (tok57 - vocab.token_mean[None, :]) / (vocab.token_std[None, :] + 1e-8)
        assign = soft_assign(z, vocab.centers, vocab.tau)
        vlad = vlad_pool(z, vocab.centers, assign)
        # Keep the spectral context channel count aligned with the fitted
        # vocabulary.  The locked representation uses K=8, but parameter
        # sensitivity also evaluates K=4/16/32.
        aall = torch.zeros((central.scale.n_edges_x, int(vocab.centers.shape[0])), device=device, dtype=assign.dtype)
        aall[central.scale.cross_rows_x] = assign
        context = spectral_signal_response(aall, eig_results[1][1], eig_results[1][2], cfg.response_times, modes=cfg.response_modes)
        rep = torch.cat([torch.cat(global_blocks), central.scale.chem15_cross.mean(dim=0), chem_blocks[1], tok114, vlad, context])
        return assign, vlad, context, rep
    stage_ms[STAGES[9]], (_, vlad, context, rep) = _timed(device, vlad_context)
    # Compute the expected readout width from the actual concatenated blocks.
    # The context response width is not simply 2*K for every K because the
    # spectral response operator has its own mode/time layout.  Deriving it
    # from the block tensors keeps the shape guard valid for K=4/16/32 and for
    # token-component ablations without weakening the check.
    base_dim = sum(int(block.numel()) for block in global_blocks)
    base_dim += int(central.scale.chem15_cross.mean(dim=0).numel())
    base_dim += int(chem_blocks[1].numel()) + int(tok114.numel())
    expected_dim = base_dim + int(vlad.numel()) + int(context.numel())
    if int(rep.numel()) != expected_dim:
        raise RuntimeError(f"PGL-Rep output dimension changed for {ablation}: {rep.numel()} != {expected_dim}")
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
    result = {"stage_ms": stage_ms, "meta": meta, "peak_mb": peak_mb, "output_dim": int(rep.numel())}
    if return_features:
        # Feature extraction is opt-in so the profiler keeps its lightweight
        # timing-only behavior.  CPU copies make the caller independent of
        # the GPU allocator between complexes.
        result["tok57"] = tok57.detach().float().cpu().numpy()
        result["rep1137"] = rep.detach().float().cpu().numpy()
        result["diagnostic_gauss"] = [
            pair[0].detach().float().cpu().numpy() for pair in gauss
        ]
        result["diagnostic_eigenvalues"] = [
            pair[1].detach().float().cpu().numpy() for pair in eig_results
        ]
        result["diagnostic_scales"] = [
            float(pair[3].detach().float().cpu()) for pair in eig_results
        ]
    return result


def _fixed_vocab(device: torch.device, cfg: PGLConfig, token_dim: int = 57) -> Vocabulary:
    centers = torch.zeros((8, token_dim), device=device, dtype=cfg.dtype)
    centers[:, :min(8, token_dim)] = torch.eye(min(8, token_dim), device=device, dtype=cfg.dtype)
    return Vocabulary(
        token_mean=torch.zeros(token_dim, device=device, dtype=cfg.dtype),
        token_std=torch.ones(token_dim, device=device, dtype=cfg.dtype),
        centers=centers,
        tau=1.0,
    )


def _sample_paths(root: Path, n: int, seed: int) -> List[Path]:
    paths = sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob("*_protein.pdb")) and list(p.glob("*_ligand.sdf")))
    if len(paths) < n:
        raise RuntimeError(f"Only {len(paths)} complete complexes found, requested {n}")
    rng = random.Random(seed)
    return sorted(rng.sample(paths, n), key=lambda p: p.name)


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def _quantile(values: Iterable[float], q: float) -> float:
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q, method="linear"))


def _run_backend(paths: Sequence[Path], backend: str, args) -> List[Dict[str, object]]:
    device = torch.device(backend)
    spectral_mode = os.environ.get("PGL_PROFILE_SPECTRAL_MODE", "fp64_stable")
    if spectral_mode == "float32_fast":
        cfg = PGLConfig(dtype=torch.float32, spectral_mode="float32_fast", spectral_dtype=torch.float32)
    elif spectral_mode == "fp64_stable":
        cfg = PGLConfig(dtype=torch.float32, spectral_mode="fp64_stable", spectral_dtype=torch.float64)
    else:
        raise ValueError(f"Unsupported PGL_PROFILE_SPECTRAL_MODE={spectral_mode}")
    torch.set_num_threads(args.cpu_threads)
    vocab = _fixed_vocab(device, cfg)
    rows: List[Dict[str, object]] = []
    for idx, path in enumerate(paths, 1):
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                _sync(device)
                torch.cuda.reset_peak_memory_stats(device)
            # Warm-up is deliberately a full end-to-end run on the same real complex.
            run_once(path, device, cfg, vocab)
            measurements = [run_once(path, device, cfg, vocab) for _ in range(args.repeats)]
        except Exception as exc:
            rows.append({"sample_id": path.name, "backend": backend, "status": "failed", "error": repr(exc), "formal_repeats": args.repeats, "warmup_runs": 1})
            print(f"{backend} {idx:03d}/{len(paths)} {path.name} FAILED {exc!r}", flush=True)
            continue
        row: Dict[str, object] = {"sample_id": path.name, "backend": backend, "status": "ok", "error": "", "formal_repeats": args.repeats, "warmup_runs": 1, "output_dim": measurements[-1]["output_dim"]}
        for key in ("protein_ca", "ligand_heavy", "edge_s1", "edge_s2", "edge_s3", "tri_s1", "tri_s2", "tri_s3", "dim_s1", "dim_s2", "dim_s3"):
            row[key] = measurements[-1]["meta"][key]
        for stage in STAGES:
            row[stage + "_ms"] = _median(m["stage_ms"][stage] for m in measurements)
        row["total_ms"] = _median(sum(m["stage_ms"][s] for s in STAGES) for m in measurements)
        row["gpu_peak_memory_mb"] = _median(m["peak_mb"] for m in measurements)
        for stage in STAGES:
            row[stage + "_pct"] = 100.0 * float(row[stage + "_ms"]) / max(float(row["total_ms"]), 1e-12)
        rows.append(row)
        print(f"{backend} {idx:03d}/{len(paths)} {path.name} total={row['total_ms']:.2f} ms edges={row['edge_s2']} gpu_peak={row['gpu_peak_memory_mb']:.1f} MB", flush=True)
    return rows


def _write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fields = ["sample_id", "backend", "status", "error", "formal_repeats", "warmup_runs", "output_dim", "protein_ca", "ligand_heavy", "edge_s1", "edge_s2", "edge_s3", "tri_s1", "tri_s2", "tri_s3", "dim_s1", "dim_s2", "dim_s3", "total_ms", "gpu_peak_memory_mb"]
    fields += [x + suffix for x in STAGES for suffix in ("_ms", "_pct")]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_report(rows: Sequence[Dict[str, object]], path: Path, host: str, gpu_name: str, args) -> None:
    by_backend = {b: [r for r in rows if r["backend"] == b] for b in sorted({r["backend"] for r in rows})}
    spectral_mode = os.environ.get("PGL_PROFILE_SPECTRAL_MODE", "fp64_stable")
    lines = ["# PGL-Rep GPU v1 runtime profiling", "", f"- Host: `{host}`", f"- GPU: `{gpu_name or 'not available'}`", f"- Complexes: {len({r['sample_id'] for r in rows})} random complete PDBbind v2016 complexes (seed={args.seed})", f"- Formal repeats: {args.repeats} per complex/backend; one full warm-up per complex/backend", f"- Spectral mode: `{spectral_mode}`; geometry/chemistry dtype remains float32", "- Timing: `perf_counter`; every GPU stage is bracketed by `torch.cuda.synchronize()`", ""]
    lines += ["## Aggregate stage distribution", "", "Statistics are computed over successful complex-level medians; failures remain visible in the CSV and failure-rate column.", "", "| backend | n ok / n requested | failure rate | total median (ms) | p90 (ms) | p99 (ms) | dominant stage | dominant share | GPU peak median (MB) | peak p99 (MB) |", "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for backend, rs in by_backend.items():
        ok = [r for r in rs if r.get("status") == "ok"]
        if not ok:
            lines.append(f"| {backend} | 0 / {len(rs)} | 100.0% | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        med = {s: _median(float(r[s + "_pct"]) for r in ok) for s in STAGES}
        dom = max(STAGES, key=lambda s: med[s])
        total = [float(r["total_ms"]) for r in ok]
        peak = [float(r["gpu_peak_memory_mb"]) for r in ok]
        lines.append(f"| {backend} | {len(ok)} / {len(rs)} | {100.0*(len(rs)-len(ok))/max(len(rs),1):.1f}% | {_median(total):.2f} | {_quantile(total,.90):.2f} | {_quantile(total,.99):.2f} | {dom} | {med[dom]:.1f}% | {_median(peak):.1f} | {_quantile(peak,.99):.1f} |")
    lines += ["", "## CPU/GPU comparison", ""]
    if "cpu" in by_backend and "cuda" in by_backend:
        cpu = {r["sample_id"]: r for r in by_backend["cpu"] if r.get("status") == "ok"}
        gpu = {r["sample_id"]: r for r in by_backend["cuda"] if r.get("status") == "ok"}
        paired = [float(cpu[k]["total_ms"]) / max(float(gpu[k]["total_ms"]), 1e-12) for k in cpu.keys() & gpu.keys()]
        lines.append(f"Paired median total-time speedup (CPU/GPU): **{_median(paired):.2f}x** across {len(paired)} complexes.")
        for s in ["gauss_edge_pair_integral", "MG_decomp_PGL_eig", "persistent_up_schur"]:
            sp = [float(cpu[k][s + "_ms"]) / max(float(gpu[k][s + "_ms"]), 1e-12) for k in cpu.keys() & gpu.keys()]
            lines.append(f"- `{s}` paired speedup: {_median(sp):.2f}x")
    else:
        lines.append("Only one backend completed; no paired speedup is reported.")
    lines += ["", "## Edge-count scaling", "", "The CSV records X-edge counts and matrix dimensions for all three (4→5, 5→6, 6→7) scales. The following quartile summaries use the central 5→6 matrix dimension:", "", "| backend | edge bin | n | median total (ms) | median Gauss (ms) | median eig (ms) |", "|---|---|---:|---:|---:|---:|"]
    for backend, rs in by_backend.items():
        ordered = sorted(rs, key=lambda r: int(r["edge_s2"]))
        q = max(1, len(ordered) // 4)
        for qi in range(4):
            subset = ordered[qi * q: (qi + 1) * q if qi < 3 else len(ordered)]
            if not subset:
                continue
            lines.append(f"| {backend} | Q{qi+1} ({subset[0]['edge_s2']}–{subset[-1]['edge_s2']}) | {len(subset)} | {_median(float(r['total_ms']) for r in subset):.2f} | {_median(float(r['gauss_edge_pair_integral_ms']) for r in subset):.2f} | {_median(float(r['MG_decomp_PGL_eig_ms']) for r in subset):.2f} |")
    lines += ["", "## Interpretation", "", "- The main bottleneck is the stage with the largest median share in the aggregate table; the full per-complex distribution is in `profiling_results.csv`.", "- Gauss is expected to scale approximately quadratically in the number of cross interaction edges because the existing implementation evaluates a dense pair matrix in blocks.", "- PGL eigendecomposition and the M_G eigendecomposition are dense symmetric eigensolves; their cost grows steeply with matrix dimension and should be interpreted against `dim_s2`/`dim_s3`, not only cross-edge count.", "- The next optimization target should be the empirically dominant stage above. Candidate work must preserve the current Gauss kernel, persistent Schur definition, and 1137-D concatenation; this profiling does not implement an optimization.", "", "## Reproducibility notes", "", f"- CPU threads: {args.cpu_threads}; sample selection is deterministic with seed {args.seed}.", "- Data/topology preparation uses real PDBbind protein CA coordinates, ligand heavy-atom SDF coordinates/bonds, real-coordinate pocket graphs, and deterministic 15-D contact descriptors only to supply the existing fixed-shape API.", "- Temporary extracted samples and remote benchmark scripts are removed after result transfer; the pre-existing dataset archives and unrelated processes are left untouched."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    ap.add_argument("--n-complexes", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda"])
    ap.add_argument("--host", default="remote")
    ap.add_argument("--gpu-name", default="")
    args = ap.parse_args()
    if args.repeats < 5:
        raise ValueError("Formal repeats must be at least 5")
    paths = _sample_paths(args.root, args.n_complexes, args.seed)
    rows: List[Dict[str, object]] = []
    for backend in args.backends:
        if backend == "cuda" and not torch.cuda.is_available():
            print("Skipping cuda: torch.cuda.is_available() is false", flush=True)
            continue
        rows.extend(_run_backend(paths, backend, args))
    _write_csv(rows, args.output_csv)
    gpu_name = args.gpu_name or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    _write_report(rows, args.output_report, args.host, gpu_name, args)
    print(f"Wrote {args.output_csv} and {args.output_report}", flush=True)


if __name__ == "__main__":
    main()
