import numpy as np
import torch

from pglrep import PGLConfig, PGLVocabulary
from pglrep._canonical_pipeline import ScaleInput, extract_pgl_rep_1137


def _scale(offset: int) -> ScaleInput:
    b1 = torch.tensor([[-1., 0., 0.], [1., -1., 0.], [0., 1., -1.]])
    return ScaleInput(
        B1=b1, B2p=torch.zeros((3, 0)), beta1=1,
        cross_rows_x=torch.tensor([0, 1]),
        p_start=torch.tensor([[0., 0., 0.], [0., 1., 0.]]),
        l_end=torch.tensor([[1., 0., 0.], [1., 1., 0.]]),
        endpoint_ids=torch.tensor([[offset, 10 + offset], [1 + offset, 11 + offset]]),
        chem15_cross=torch.nn.functional.one_hot(torch.tensor([0, 1]), 15).float(),
    )


def test_canonical_readout_is_1137d():
    rng = np.random.default_rng(1)
    vocab = PGLVocabulary.fit(rng.normal(size=(32, 57)).astype(np.float32))
    rep = extract_pgl_rep_1137(
        (_scale(0), _scale(20), _scale(40)),
        torch.tensor([[0., 0., 0.], [0., 1., 0.]]),
        torch.tensor([[1., 0., 0.], [1., 1., 0.]]),
        vocab.to_canonical(torch.device("cpu"), torch.float32), PGLConfig(),
    )
    assert rep.shape == (1137,)
    assert torch.isfinite(rep).all()
