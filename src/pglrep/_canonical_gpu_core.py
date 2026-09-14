"""
PGL-Rep GPU core (reference implementation skeleton)
====================================================
Heavy numerical path for the fixed PGL-Rep design:
  cross-edge Gauss kernel -> Gauss metric -> symmetric PGL operator
  -> global spectral features / local HKS-wavelet / chemical response
  -> point-cloud interaction tokens.

Combinatorial construction of X subset Y and persistent B2 is intentionally
kept as a CPU preprocessing stage (see persistent_b2_from_B2Y). This is the
right split for small/medium molecular complexes: irregular simplex bookkeeping
is CPU-friendly, while O(m^2)-O(m^3) numerical kernels are GPU-friendly.

Dependencies: numpy, scipy, torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple
import math
import os
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class PGLConfig:
    gauss_eps: float = 0.5
    gauss_lambda: float = 1.0
    gauss_quad_order: int = 4
    gauss_block: int = 256
    # Main spectral readout scales from the frozen v1 design.
    heat_times: Tuple[float, ...] = (0.25, 1.0, 4.0, 16.0)
    response_times: Tuple[float, ...] = (0.25, 1.0, 4.0)
    wavelet_times: Tuple[float, ...] = (0.25, 1.0, 4.0)
    spectral_density_bins: int = 16
    spectral_density_lo: float = -4.0
    spectral_density_hi: float = 4.0
    spectral_density_sigma: float = 0.35
    response_modes: int = 256
    zero_tol: float = 1e-7
    # Geometry/chemistry tensors may remain float32. The canonical spectral
    # algebra uses an explicit float64 work dtype; float32 is retained only
    # for a named fast diagnostic/legacy run.
    dtype: torch.dtype = torch.float32
    spectral_mode: str = "fp64_stable"
    spectral_dtype: torch.dtype = torch.float64


def _legendre01(order: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    x, w = np.polynomial.legendre.leggauss(order)
    # [-1, 1] -> [0, 1]
    x = (x + 1.0) * 0.5
    w = w * 0.5
    return (
        torch.as_tensor(x, device=device, dtype=dtype),
        torch.as_tensor(w, device=device, dtype=dtype),
    )


@torch.no_grad()
def gauss_linking_matrix_gpu(
    starts: Tensor,
    ends: Tensor,
    endpoint_ids: Tensor,
    chem_q: Tensor,
    cfg: PGLConfig,
    *,
    return_signed: bool = False,
):
    """Dense absolute, chemistry-weighted Gauss matrix for PL interaction edges.

    Parameters
    ----------
    starts, ends: [m, 3]
        Oriented interaction edges, always Protein -> Ligand.
    endpoint_ids: [m, 2] int64
        Global vertex IDs. Pairs sharing an endpoint are forced to zero.
    chem_q: [m, C]
        Fixed chemical multi-hot edge descriptors.

    Notes
    -----
    For straight segments gamma_a(s)=a+s*u and gamma_b(t)=c+t*v,
      (gamma_a-gamma_b) dot (u x v)
    is independent of s,t because u dot (u x v)=v dot (u x v)=0.
    Therefore only the denominator needs quadrature. This substantially reduces
    arithmetic and makes the kernel GPU-friendly.
    """
    assert starts.shape == ends.shape and starts.ndim == 2 and starts.shape[1] == 3
    device, dtype = starts.device, starts.dtype
    m = starts.shape[0]
    if m == 0:
        return torch.zeros((0, 0), device=device, dtype=dtype)

    # Gauss linking is translation-invariant.  Centering the coordinates
    # before float32 arithmetic removes large common offsets and prevents
    # cancellation from magnifying harmless file-rounding differences.
    origin = torch.cat((starts, ends), dim=0).mean(dim=0)
    starts = starts - origin
    ends = ends - origin

    qx, qw = _legendre01(cfg.gauss_quad_order, device, dtype)
    dirs = ends - starts
    W = torch.zeros((m, m), device=device, dtype=dtype)
    S = torch.zeros((m, m), device=device, dtype=dtype) if return_signed else None
    block = cfg.gauss_block
    inv4pi = 1.0 / (4.0 * math.pi)

    qnorm = (chem_q * chem_q).sum(dim=1)

    for i0 in range(0, m, block):
        i1 = min(m, i0 + block)
        ai = starts[i0:i1]
        ui = dirs[i0:i1]
        idi = endpoint_ids[i0:i1]
        qi = chem_q[i0:i1]
        qni = qnorm[i0:i1]

        for j0 in range(i0, m, block):
            j1 = min(m, j0 + block)
            aj = starts[j0:j1]
            uj = dirs[j0:j1]
            idj = endpoint_ids[j0:j1]
            qj = chem_q[j0:j1]
            qnj = qnorm[j0:j1]

            # [bi,bj,3]
            r0 = ai[:, None, :] - aj[None, :, :]
            cross = torch.cross(ui[:, None, :].expand(-1, j1-j0, -1),
                                uj[None, :, :].expand(i1-i0, -1, -1), dim=-1)
            numerator = (r0 * cross).sum(dim=-1)

            integ = torch.zeros_like(numerator)
            # Small fixed loop (4x4 by default); each body is a large fused tensor op.
            for a in range(cfg.gauss_quad_order):
                for b in range(cfg.gauss_quad_order):
                    rr = r0 + qx[a] * ui[:, None, :] - qx[b] * uj[None, :, :]
                    denom = (rr.square().sum(dim=-1) + cfg.gauss_eps ** 2).pow(1.5)
                    integ.add_((qw[a] * qw[b]) / denom)

            G = (numerator * integ * inv4pi).abs()

            # exp(-||q_i-q_j||^2/4)
            dist2 = (qni[:, None] + qnj[None, :] - 2.0 * (qi @ qj.T)).clamp_min_(0.0)
            Kchem = torch.exp(-0.25 * dist2)
            Wij = G * Kchem

            # shared endpoint mask
            shared = ((idi[:, None, 0] == idj[None, :, 0]) |
                      (idi[:, None, 0] == idj[None, :, 1]) |
                      (idi[:, None, 1] == idj[None, :, 0]) |
                      (idi[:, None, 1] == idj[None, :, 1]))
            Wij.masked_fill_(shared, 0.0)
            if i0 == j0:
                Wij.fill_diagonal_(0.0)
                # Enforce exact symmetry within a diagonal block.
                Wij = 0.5 * (Wij + Wij.T)

            W[i0:i1, j0:j1] = Wij
            if return_signed:
                Sij = (numerator * integ * inv4pi) * Kchem
                Sij.masked_fill_(shared, 0.0)
                if i0 == j0:
                    Sij.fill_diagonal_(0.0)
                    Sij = 0.5 * (Sij + Sij.T)
                S[i0:i1, j0:j1] = Sij
            if j0 != i0:
                W[j0:j1, i0:i1] = Wij.T
                if return_signed:
                    S[j0:j1, i0:i1] = Sij.T

    return (W, S) if return_signed else W


@torch.no_grad()
def embed_cross_gauss_into_all_edges(
    n_edges_x: int,
    cross_rows_x: Tensor,
    W_cross: Tensor,
) -> Tensor:
    """Embed cross-edge Gauss coupling into the full C1(X) edge basis."""
    device, dtype = W_cross.device, W_cross.dtype
    W = torch.zeros((n_edges_x, n_edges_x), device=device, dtype=dtype)
    W[cross_rows_x[:, None], cross_rows_x[None, :]] = W_cross
    return W


@torch.no_grad()
def gauss_metric_dense(W: Tensor, lam: float = 1.0, eps: float = 1e-12) -> Tuple[Tensor, Tensor, Tensor]:
    """M_G = I + lam * L_G/(2*dmax+eps), guaranteed SPD for lam>=0."""
    d = W.sum(dim=1)
    LG = torch.diag(d) - W
    dmax = d.max() if d.numel() else torch.tensor(0.0, device=W.device, dtype=W.dtype)
    scale = 2.0 * dmax + eps
    Lbar = LG / scale if float(dmax) > 0.0 else LG
    M = torch.eye(W.shape[0], device=W.device, dtype=W.dtype) + lam * Lbar
    return M, LG, d


def _dense(x: Tensor) -> Tensor:
    return x.to_dense() if x.is_sparse else x


@torch.no_grad()
def pgl_symmetric_dense(
    B1: Tensor,
    B2p: Tensor,
    M: Tensor,
    *,
    eig_floor: float = 1e-8,
    spectral_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build the principal symmetric representative of the metric PGL.

    A = B1^T B1 + M B2 B2^T M
    Delta_tilde = M^{-1/2} A M^{-1/2}

    Returns
    -------
    Delta_hat : mean-positive-eigenvalue normalized symmetric PGL
    evals_hat, evecs : eigensystem of Delta_hat
    scale : positive-eigenvalue mean of unnormalized Delta_tilde
    """
    work_dtype = spectral_dtype or M.dtype
    B1d = _dense(B1).to(dtype=work_dtype)
    B2d = _dense(B2p).to(dtype=work_dtype)
    M = M.to(dtype=work_dtype)

    # Principal M^{-1/2}; this keeps local edge coordinates well-defined.
    me, U = torch.linalg.eigh(M)
    me = me.clamp_min(eig_floor)
    Minvhalf = (U * me.rsqrt()[None, :]) @ U.T

    down = B1d.T @ B1d
    MB2 = M @ B2d
    A = down + MB2 @ MB2.T
    Delta = Minvhalf @ A @ Minvhalf
    Delta = 0.5 * (Delta + Delta.T)

    evals, evecs = torch.linalg.eigh(Delta)
    pos = evals > eig_floor
    if bool(pos.any()):
        scale = evals[pos].mean()
    else:
        scale = torch.tensor(1.0, device=evals.device, dtype=evals.dtype)
    evals_hat = evals / scale
    Delta_hat = (evecs * evals_hat[None, :]) @ evecs.T
    Delta_hat = 0.5 * (Delta_hat + Delta_hat.T)
    return Delta_hat, evals_hat, evecs, scale


@torch.no_grad()
def global_spectral_features(
    evals_hat: Tensor,
    scale: Tensor,
    beta1: int,
    cfg: PGLConfig,
) -> Tensor:
    """28-D global block for one filtration pair."""
    pos = evals_hat > cfg.zero_tol
    ep = evals_hat[pos]
    device, dtype = evals_hat.device, evals_hat.dtype
    if ep.numel() == 0:
        ep = torch.zeros(1, device=device, dtype=dtype)

    ep_sorted = torch.sort(ep).values
    mu1 = ep_sorted[0]
    gap2 = ep_sorted[1] - ep_sorted[0] if ep_sorted.numel() > 1 else torch.tensor(0., device=device, dtype=dtype)
    scalars = torch.stack([
        torch.tensor(float(beta1), device=device, dtype=dtype),
        torch.tensor(float(pos.sum()), device=device, dtype=dtype),
        torch.log1p(scale),
        mu1,
        gap2,
        ep_sorted[-1],
        torch.log1p(ep).mean(),
        (1.0 / (1.0 + ep)).mean(),
    ])

    x = torch.log(ep + 1e-8)
    centers = torch.linspace(cfg.spectral_density_lo, cfg.spectral_density_hi,
                             cfg.spectral_density_bins, device=device, dtype=dtype)
    rho = torch.exp(-0.5 * ((x[:, None] - centers[None, :]) / cfg.spectral_density_sigma) ** 2).mean(dim=0)
    heat = torch.stack([torch.exp(-t * evals_hat.clamp_min(0)).mean() for t in cfg.heat_times])
    return torch.cat([scalars, rho, heat], dim=0)


@torch.no_grad()
def local_hks_wavelet_from_eigs(
    evals_hat: Tensor,
    evecs: Tensor,
    cfg: PGLConfig,
) -> Tensor:
    """[n_edges, 7]: 4 HKS + 3 diffusion-band energies."""
    U2 = evecs.square()
    l = evals_hat.clamp_min(0)
    feats = []
    for t in cfg.heat_times:
        feats.append(U2 @ torch.exp(-t * l))
    for t in cfg.wavelet_times:
        psi = torch.exp(-t * l) - torch.exp(-2.0 * t * l)
        # diag(Psi^2) = sum_k psi(lambda_k)^2 * u_k(e)^2
        feats.append(U2 @ psi.square())
    return torch.stack(feats, dim=1)


@torch.no_grad()
def _spectral_signal_gram_base(
    B: Tensor,
    evecs: Tensor,
    modes: int,
) -> Tuple[Tensor, Tensor]:
    """Precompute S_c = U^T diag(B_c) U for K⊙K response readout.

    Uses the lowest `modes` PGL modes. Returns S [C,k,k] and U_k.
    """
    n, C = B.shape
    k = min(modes, evecs.shape[1])
    U = evecs[:, :k]
    S = []
    for c in range(C):
        BU = B[:, c:c+1].to(dtype=U.dtype) * U
        S.append(U.T @ BU)
    return torch.stack(S, dim=0), U


@torch.no_grad()
def spectral_signal_response(
    B: Tensor,
    evals_hat: Tensor,
    evecs: Tensor,
    times: Sequence[float],
    modes: int = 256,
    eps: float = 1e-8,
) -> Tensor:
    """Compute normalized B^T (K_t⊙K_t) B without forming dense K_t.

    Identity used:
      K=U diag(d) U^T
      C_cd = sum_ab d_a d_b S_c[a,b] S_d[a,b],
      S_c = U^T diag(B_c) U.

    Returns concatenated upper triangles for each time.
    """
    B = B.to(dtype=evecs.dtype)
    S, _ = _spectral_signal_gram_base(B, evecs, modes)
    k = S.shape[-1]
    l = evals_hat[:k].clamp_min(0)
    norm = torch.sqrt((B.square().sum(dim=0)).clamp_min(eps))
    C = B.shape[1]
    iu = torch.triu_indices(C, C, device=B.device)
    out = []
    for t in times:
        d = torch.exp(-t * l)
        root = torch.sqrt(d)
        R = S * root[None, :, None] * root[None, None, :]
        R = R.flatten(1)
        G = R @ R.T
        G = G / (norm[:, None] * norm[None, :])
        out.append(G[iu[0], iu[1]])
    return torch.cat(out, dim=0)


@torch.no_grad()
def local_gauss_context(W_cross: Tensor, G_signed_cross: Optional[Tensor] = None, eps: float = 1e-8) -> Tensor:
    """[m,3]: absolute degree, signed score (optional), neighborhood entropy."""
    g = W_cross.sum(dim=1)
    p = W_cross / (g[:, None] + eps)
    ent = -(p * torch.log(p + eps)).sum(dim=1)
    if G_signed_cross is None:
        signed = torch.zeros_like(g)
    else:
        signed = G_signed_cross.sum(dim=1) / (g + eps)
    return torch.stack([g, signed, ent], dim=1)


@torch.no_grad()
def _shape_features(mid: Tensor, atom_xyz: Tensor, radius: float = 6.0, eps: float = 1e-8) -> Tuple[Tensor, Tensor, Tensor]:
    """Per-midpoint counts and covariance eigen-shape features."""
    if atom_xyz.numel() == 0:
        m = mid.shape[0]
        z = torch.zeros((m,), device=mid.device, dtype=mid.dtype)
        return z, torch.zeros((m,3), device=mid.device, dtype=mid.dtype), torch.empty((m,0), device=mid.device, dtype=mid.dtype)
    delta = atom_xyz[None, :, :] - mid[:, None, :]
    dist = torch.linalg.vector_norm(delta, dim=-1)
    mask = (dist <= radius).to(mid.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    cov = torch.einsum('eni,enj,en->eij', delta, delta, mask) / count[:, None, None]
    vals = torch.linalg.eigvalsh(cov).flip(-1).clamp_min(0)
    g1, g2, g3 = vals.unbind(dim=1)
    shape = torch.stack([(g1-g2)/(g1+eps), (g2-g3)/(g1+eps), g3/(g1+eps)], dim=1)
    return mask.sum(dim=1), shape, dist


@torch.no_grad()
def pointcloud32(
    p_start: Tensor,
    l_end: Tensor,
    protein_xyz: Tensor,
    ligand_xyz: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Fixed 32-D local point-cloud geometry block per interaction edge."""
    mid = 0.5 * (p_start + l_end)
    axis_vec = l_end - p_start
    contact_d = torch.linalg.vector_norm(axis_vec, dim=1, keepdim=True)
    axis = axis_vec / (contact_d + eps)

    def one_side(xyz: Tensor):
        delta = xyz[None, :, :] - mid[:, None, :]
        dist = torch.linalg.vector_norm(delta, dim=-1)
        mask4 = (dist <= 4.0).to(mid.dtype)
        mask6 = (dist <= 6.0).to(mid.dtype)
        n4 = mask4.sum(dim=1, keepdim=True)
        n6 = mask6.sum(dim=1, keepdim=True)

        denom = n6.clamp_min(1.0)
        cov = torch.einsum('eni,enj,en->eij', delta, delta, mask6) / denom[:, :, None]
        vals = torch.linalg.eigvalsh(cov).flip(-1).clamp_min(0)
        g1, g2, g3 = vals.unbind(1)
        shape = torch.stack([(g1-g2)/(g1+eps), (g2-g3)/(g1+eps), g3/(g1+eps)], dim=1)

        centers = torch.arange(1, 7, device=xyz.device, dtype=xyz.dtype)
        rbf = torch.exp(-0.5 * ((dist[:, :, None] - centers[None, None, :]) / 0.75) ** 2)
        radial = (rbf * mask6[:, :, None]).sum(dim=1) / denom

        unit = delta / (dist[:, :, None] + eps)
        cos = (unit * axis[:, None, :]).sum(dim=-1).clamp(-1, 1)
        # Legendre P1..P4
        P1 = cos
        P2 = 0.5 * (3*cos**2 - 1)
        P3 = 0.5 * (5*cos**3 - 3*cos)
        P4 = (35*cos**4 - 30*cos**2 + 3) / 8.0
        ang = torch.stack([P1,P2,P3,P4], dim=-1)
        angular = (ang * mask6[:, :, None]).sum(dim=1) / denom
        return n4, n6, shape, radial, angular

    p4,p6,ps,pr,pa = one_side(protein_xyz)
    l4,l6,ls,lr,la = one_side(ligand_xyz)
    burial = (p6-l6)/(p6+l6+eps)
    # 1 + 4 counts + 6 shape + 12 radial + 8 angular + 1 burial = 32
    return torch.cat([contact_d, p4,l4,p6,l6, ps,ls, pr,lr, pa,la, burial], dim=1)


@torch.no_grad()
def build_token57(
    chem15_cross: Tensor,
    local_spec_all_edges: Tensor,
    cross_rows_x: Tensor,
    point32_cross: Tensor,
    gauss3_cross: Tensor,
) -> Tensor:
    spec7 = local_spec_all_edges[cross_rows_x]
    z = torch.cat([chem15_cross, spec7, point32_cross, gauss3_cross], dim=1)
    assert z.shape[1] == 57
    return z


@torch.no_grad()
def token_mean_std(token57: Tensor) -> Tensor:
    if token57.shape[0] == 0:
        return torch.zeros(114, device=token57.device, dtype=token57.dtype)
    return torch.cat([token57.mean(dim=0), token57.std(dim=0, unbiased=False)], dim=0)


@torch.no_grad()
def soft_assign(tokens_z: Tensor, centers: Tensor, tau: float, eps: float = 1e-8) -> Tensor:
    # Stable spectral blocks are float64 while geometry/chemistry storage and
    # the persisted vocabulary remain float32.  Keep the distance algebra in
    # the token dtype so the canonical path is explicit and cdist-safe.
    centers = centers.to(dtype=tokens_z.dtype)
    dist2 = torch.cdist(tokens_z, centers).square()
    return torch.softmax(-dist2 / (2.0 * max(tau, eps)), dim=1)


@torch.no_grad()
def vlad_pool(tokens_z: Tensor, centers: Tensor, assign: Tensor, eps: float = 1e-8) -> Tensor:
    # [K,N,1] * [K,N,D]
    centers = centers.to(dtype=tokens_z.dtype)
    residual = tokens_z[:, None, :] - centers[None, :, :]
    V = (assign[:, :, None] * residual).sum(dim=0)
    V = V / (torch.linalg.vector_norm(V, dim=1, keepdim=True) + eps)
    out = V.flatten()
    return out / (torch.linalg.vector_norm(out) + eps)


# ----------------------------- CPU combinatorial oracle -----------------------------

def persistent_b2_from_B2Y(
    B2Y: sp.spmatrix,
    n_edges_x: int,
    rcond: Optional[float] = None,
) -> np.ndarray:
    """Exact persistent B2 by orthonormal nullspace construction.

    B2Y edge rows must be ordered with all C1(X) edges first, then Y\\X edges.
    If B_out Z=0 and Z has orthonormal columns, B2p=B_X Z is the persistent
    boundary in an orthonormal basis of C2^{X,Y}.
    """
    B2Y = B2Y.tocsr()
    BX = B2Y[:n_edges_x, :].toarray().astype(np.float64, copy=False)
    Bout = B2Y[n_edges_x:, :].toarray().astype(np.float64, copy=False)
    if Bout.shape[0] == 0:
        Z = np.eye(B2Y.shape[1], dtype=np.float64)
    else:
        Z = scipy.linalg.null_space(Bout, rcond=rcond)
    return BX @ Z


def persistent_beta1(B1: sp.spmatrix, B2p: np.ndarray, tol: Optional[float] = None) -> int:
    """beta1 = dim ker(B1) - rank(B2p), with float64 rank oracle."""
    B1d = B1.toarray().astype(np.float64, copy=False)
    r1 = np.linalg.matrix_rank(B1d, tol=tol)
    r2 = np.linalg.matrix_rank(B2p, tol=tol)
    return int(B1d.shape[1] - r1 - r2)


# ----------------------------- Suggested orchestration -----------------------------

@torch.no_grad()
def extract_core_blocks_one_scale(
    B1: Tensor,
    B2p: Tensor,
    W_all_edges: Tensor,
    F15_all_edges: Tensor,
    beta1: int,
    cfg: PGLConfig,
) -> Dict[str, Tensor]:
    """Numerical spectral blocks for one X->Y pair.

    For the fixed v1 representation, use this for 4->5, 5->6, 6->7 to form
    global features; use the 5->6 result for local/chemical/token readout.
    """
    M, LG, gdegree = gauss_metric_dense(W_all_edges, cfg.gauss_lambda)
    spectral_dtype = cfg.spectral_dtype if cfg.spectral_mode == "fp64_stable" else cfg.dtype
    Delta_hat, ev, U, scale = pgl_symmetric_dense(
        B1, B2p, M, spectral_dtype=spectral_dtype
    )
    global28 = global_spectral_features(ev, scale, beta1, cfg)
    local7 = local_hks_wavelet_from_eigs(ev, U, cfg)
    chem360 = spectral_signal_response(F15_all_edges, ev, U, cfg.response_times,
                                       modes=cfg.response_modes)
    return {
        'M': M,
        'LG': LG,
        'gauss_degree': gdegree,
        'Delta_hat': Delta_hat,
        'evals_hat': ev,
        'evecs': U,
        'global28': global28,
        'local7': local7,
        'chem_response360': chem360,
    }


if __name__ == '__main__':
    print('PGL-Rep GPU core loaded. Integrate with your molecular parser / simplex builder.')

# ----------------------------- GPU persistent-up backend -----------------------------

@torch.no_grad()
def up_laplacian_from_triangles_gpu(
    triangle_edge_rows: Tensor,
    triangle_edge_signs: Tensor,
    n_edges_y: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build L_up^Y = B2^Y (B2^Y)^T using 9 scatter-adds per triangle.

    triangle_edge_rows: [T,3] edge row IDs of each oriented triangle boundary.
    triangle_edge_signs: [T,3] entries in {+1,-1}.
    This avoids materializing the very sparse B2 matrix on GPU.
    """
    device = triangle_edge_rows.device
    L = torch.zeros((n_edges_y, n_edges_y), device=device, dtype=dtype)
    if triangle_edge_rows.numel() == 0:
        return L
    rows = triangle_edge_rows.long()
    s = triangle_edge_signs.to(dtype)
    for a in range(3):
        for b in range(3):
            ia = rows[:, a]
            ib = rows[:, b]
            val = s[:, a] * s[:, b]
            L.index_put_((ia, ib), val, accumulate=True)
    return 0.5 * (L + L.T)


@torch.no_grad()
def persistent_up_schur_dense_gpu(
    LupY: Tensor,
    n_edges_x: int,
    rtol: float = 1e-6,
) -> Tensor:
    """Exact (up to pseudoinverse tolerance) persistent up-Laplacian by Schur complement.

    Edges of Y must be ordered with X edges first. For
      LupY = [[A, C], [C^T, D]],
    returns H_up^{X,Y} = A - C D^dagger C^T.
    """
    A = LupY[:n_edges_x, :n_edges_x]
    if n_edges_x == LupY.shape[0]:
        return 0.5 * (A + A.T)
    C = LupY[:n_edges_x, n_edges_x:]
    D = LupY[n_edges_x:, n_edges_x:]
    if D.numel() == 0:
        return 0.5 * (A + A.T)
    de, U = torch.linalg.eigh(0.5 * (D + D.T))
    cutoff = rtol * de.abs().max().clamp_min(torch.finfo(de.dtype).eps)
    inv = torch.where(de > cutoff, de.reciprocal(), torch.zeros_like(de))
    Dpinv = (U * inv[None, :]) @ U.T
    H = A - C @ Dpinv @ C.T
    # Small numerical negatives can arise from float32 Schur subtraction.
    return 0.5 * (H + H.T)


@torch.no_grad()
def pgl_symmetric_dense_from_persistent_up(
    B1: Tensor,
    Hup: Tensor,
    M: Tensor,
    *,
    eig_floor: float = 1e-8,
    spectral_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Same PGL symmetric representative, using Hup=B2p B2p^T directly.

    This is the preferred GPU path because Hup can be constructed by the Schur
    complement without an explicit null-space basis B2p.
    """
    # Keep the canonical spectral operator and eigendecomposition in the
    # requested work dtype. The caller may explicitly request source dtype for
    # a named fast diagnostic.
    source_dtype = M.dtype
    if spectral_dtype is None:
        spectral_dtype = torch.float64 if os.environ.get("PGL_SPECTRAL_FP64", "0") == "1" else source_dtype
    work_dtype = spectral_dtype
    B1d = _dense(B1).to(dtype=work_dtype)
    Hup = Hup.to(dtype=work_dtype)
    M = M.to(dtype=work_dtype)
    me, U = torch.linalg.eigh(M)
    me = me.clamp_min(eig_floor)
    Minvhalf = (U * me.rsqrt()[None, :]) @ U.T
    down = B1d.T @ B1d
    A = down + M @ Hup @ M
    Delta = Minvhalf @ A @ Minvhalf
    Delta = 0.5 * (Delta + Delta.T)
    evals, evecs = torch.linalg.eigh(Delta)
    pos = evals > eig_floor
    scale = evals[pos].mean() if bool(pos.any()) else torch.tensor(1.0, device=evals.device, dtype=evals.dtype)
    evals_hat = evals / scale
    Delta_hat = (evecs * evals_hat[None, :]) @ evecs.T
    return 0.5 * (Delta_hat + Delta_hat.T), evals_hat, evecs, scale
