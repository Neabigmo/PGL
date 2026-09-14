import numpy as np
import pytest
import torch

from pglrep import PGLConfig, PGLVocabulary
from pglrep._canonical_gpu_core import (
    gauss_linking_matrix_gpu, gauss_metric_dense, pgl_symmetric_dense,
    persistent_b2_from_B2Y,
)


def test_gauss_translation_and_orientation_convention():
    cfg = PGLConfig()
    starts = torch.tensor([[0., 0., 0.], [0., 1., 0.]])
    ends = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
    ids = torch.tensor([[0, 2], [1, 3]])
    q = torch.zeros((2, 15))
    a = gauss_linking_matrix_gpu(starts, ends, ids, q, cfg)
    shifted = gauss_linking_matrix_gpu(starts + 1000, ends + 1000, ids, q, cfg)
    assert torch.allclose(a, shifted, atol=1e-6)
    assert torch.allclose(a, a.T, atol=1e-7)
    assert torch.diagonal(a).eq(0).all()


def test_gauss_metric_is_spd_and_laplacian_rows_sum_zero():
    w = torch.tensor([[0., 1., 2.], [1., 0., .5], [2., .5, 0.]])
    m, lg, degree = gauss_metric_dense(w)
    assert torch.allclose(lg.sum(dim=1), torch.zeros(3))
    assert torch.linalg.eigvalsh(m).min() > 0
    assert torch.all(degree >= 0)


def test_persistent_boundary_nullspace():
    import scipy.sparse as sp
    b2y = sp.csr_matrix(np.array([[1.], [-1.], [0.], [0.]]))
    out = persistent_b2_from_B2Y(b2y, 2)
    assert out.shape == (2, 1)


def test_spectrum_nonnegative_and_normalized():
    b1 = torch.tensor([[-1., 0.], [1., -1.], [0., 1.]])
    b2 = torch.zeros((2, 0))
    m = torch.eye(2)
    _, ev, _, _ = pgl_symmetric_dense(b1, b2, m)
    assert float(ev.min()) >= -1e-7
    assert float(ev[ev > 1e-8].mean()) == pytest.approx(1.0, abs=1e-6)


def test_vocabulary_fit_save_load(tmp_path):
    rng = np.random.default_rng(0)
    vocab = PGLVocabulary.fit(rng.normal(size=(32, 57)).astype(np.float32))
    path = tmp_path / "vocab.npz"
    vocab.save(path)
    loaded = PGLVocabulary.load(path)
    np.testing.assert_allclose(vocab.centers, loaded.centers)
    assert loaded.tau > 0
