"""End-to-end orchestration for the fixed 1137-D PGL-Rep v1 representation.

This file assumes molecular parsing / filtration / persistent B2 construction
have already produced ScaleInput objects. See pgl_rep_gpu_core.py for the heavy
GPU kernels and CPU persistent-boundary oracle.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import json
import numpy as np
import torch

from ._canonical_gpu_core import (
    PGLConfig, gauss_linking_matrix_gpu, embed_cross_gauss_into_all_edges,
    extract_core_blocks_one_scale, pointcloud32, local_gauss_context,
    build_token57, token_mean_std, soft_assign, vlad_pool,
    spectral_signal_response,
)

Tensor = torch.Tensor


@dataclass
class ScaleInput:
    # C1(X) / persistent C2(X,Y)
    B1: Tensor                 # [n_vertices_x, n_edges_x]
    B2p: Tensor                # [n_edges_x, n_persistent_2chains]
    beta1: int

    # PL interaction edges present in X, ordered Protein -> Ligand
    cross_rows_x: Tensor       # [m] row indices into C1(X)
    p_start: Tensor            # [m,3]
    l_end: Tensor              # [m,3]
    endpoint_ids: Tensor       # [m,2] global vertex IDs
    chem15_cross: Tensor       # [m,15]

    @property
    def n_edges_x(self) -> int:
        return int(self.B1.shape[1])


@dataclass
class Vocabulary:
    token_mean: Tensor         # [57]
    token_std: Tensor          # [57]
    centers: Tensor            # [8,57], learned on standardized train tokens only
    tau: float


def load_trained_vocab(path: Path, device: torch.device, dtype: torch.dtype | None = None) -> Vocabulary:
    """Load a persisted canonical vocabulary and reject tau-less legacy files."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"token_mean", "token_std", "centers", "tau"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(
                f"Vocabulary artifact {path} is missing {missing}; refusing silent tau fallback"
            )
        tau = float(np.asarray(data["tau"]).reshape(()))
        if not np.isfinite(tau) or tau <= 0:
            raise RuntimeError(f"Invalid trained vocabulary tau={tau} in {path}")
        target_dtype = dtype or torch.float32
        vocab = Vocabulary(
            token_mean=torch.as_tensor(data["token_mean"], device=device, dtype=target_dtype),
            token_std=torch.as_tensor(data["token_std"], device=device, dtype=target_dtype),
            centers=torch.as_tensor(data["centers"], device=device, dtype=target_dtype),
            tau=tau,
        )
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if float(meta.get("tau", tau)) != tau:
            raise RuntimeError(f"Vocabulary npz/json tau disagreement for {path}")
        if meta.get("implementation_revision") not in (None, "canonical"):
            raise RuntimeError(f"Unexpected vocabulary implementation revision in {meta_path}")
    return vocab


@torch.no_grad()
def extract_pgl_rep_1137(
    scales: Tuple[ScaleInput, ScaleInput, ScaleInput],
    protein_xyz: Tensor,
    ligand_xyz: Tensor,
    vocab: Vocabulary,
    cfg: PGLConfig = PGLConfig(),
) -> Tensor:
    """Return fixed 1137-D PGL-Rep for (4->5, 5->6, 6->7) inputs.

    The middle scale (5->6) is the central local/token scale.
    """
    global_blocks = []
    central = None

    for si, s in enumerate(scales):
        Wc, Sc = gauss_linking_matrix_gpu(
            s.p_start, s.l_end, s.endpoint_ids, s.chem15_cross, cfg,
            return_signed=True,
        )
        Wall = embed_cross_gauss_into_all_edges(s.n_edges_x, s.cross_rows_x, Wc)
        Fall = torch.zeros((s.n_edges_x, 15), device=s.B1.device, dtype=s.B1.dtype)
        Fall[s.cross_rows_x] = s.chem15_cross

        blocks = extract_core_blocks_one_scale(
            s.B1, s.B2p, Wall, Fall, s.beta1, cfg
        )
        global_blocks.append(blocks['global28'])
        if si == 1:
            central = (s, Wc, Sc, Fall, blocks)

    assert central is not None
    s, Wc, Sc, Fall, blocks = central

    # 84 global dims
    global84 = torch.cat(global_blocks, dim=0)

    # 15 chemistry abundance dims
    if s.chem15_cross.shape[0] > 0:
        chem15 = s.chem15_cross.mean(dim=0)
    else:
        chem15 = torch.zeros(15, device=s.B1.device, dtype=s.B1.dtype)

    # 360 chemical spectral-response dims
    chem360 = blocks['chem_response360']

    # 57-D interaction tokens at central scale
    pc32 = pointcloud32(s.p_start, s.l_end, protein_xyz, ligand_xyz)
    g3 = local_gauss_context(Wc, Sc)
    tok57 = build_token57(
        s.chem15_cross, blocks['local7'], s.cross_rows_x, pc32, g3
    )

    # 114 raw token distribution dims
    tok114 = token_mean_std(tok57)

    # Standardized tokens for vocabulary/VLAD only
    z = (tok57 - vocab.token_mean[None, :]) / (vocab.token_std[None, :] + 1e-8)
    A = soft_assign(z, vocab.centers, vocab.tau)  # [m,8]

    # 456 VLAD dims
    vlad456 = vlad_pool(z, vocab.centers, A)

    # 108 PGL token-context dims
    # Assignment is promoted to the explicit spectral work dtype by the
    # canonical response path; keep the scatter destination type identical.
    Aall = torch.zeros((s.n_edges_x, 8), device=s.B1.device, dtype=A.dtype)
    Aall[s.cross_rows_x] = A
    context108 = spectral_signal_response(
        Aall, blocks['evals_hat'], blocks['evecs'], cfg.response_times,
        modes=cfg.response_modes,
    )

    rep = torch.cat([global84, chem15, chem360, tok114, vlad456, context108], dim=0)
    if rep.numel() != 1137:
        raise RuntimeError(f'Expected 1137-D PGL-Rep, got {rep.numel()}')
    return rep
