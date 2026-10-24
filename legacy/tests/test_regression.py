"""Optional legal-fixture regression against the extracted canonical harness.

Set PGLREP_REGRESSION_CASE to a directory containing protein.pdb and ligand.sdf
and PGLREP_REGRESSION_VOCAB to a canonical train vocabulary to run it. No
database structures are bundled with the package.
"""
import os
from pathlib import Path
import numpy as np
import pytest

from pglrep import PGLConfig, PGLRep, PGLVocabulary


@pytest.mark.slow
def test_package_fixture_is_deterministic():
    case = os.getenv("PGLREP_REGRESSION_CASE")
    vocab = os.getenv("PGLREP_REGRESSION_VOCAB")
    if not case or not vocab:
        pytest.skip("requires a user-provided legal fixture and train-only vocabulary")
    root = Path(case)
    model = PGLRep(PGLConfig(), PGLVocabulary.load(vocab), device="cpu")
    a = model.from_complex(root / "protein.pdb", root / "ligand.sdf").vector
    b = model.from_complex(root / "protein.pdb", root / "ligand.sdf").vector
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
