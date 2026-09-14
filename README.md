# PGL-Rep

PGL-Rep (Persistent Gauss-Linking Laplacians) is a fixed, explicit protein–ligand representation. It combines persistent interaction complexes, a regularized chemistry-weighted Gauss relation, and a generalized spectral readout into a 1,137-dimensional vector.

## Status

Version `0.1.0` is a paper-compatible source extraction under local validation. The repository does not currently declare a software license; users should obtain permission before redistribution or commercial use.

## Installation

```bash
python -m pip install .
```

## Quick start

The 1137-D readout requires a vocabulary fitted on training tokens only:

```python
from pglrep import PGLConfig, PGLRep, PGLVocabulary

vocab = PGLVocabulary.load("train_vocab.npz")
result = PGLRep(PGLConfig(), vocab, device="cpu").from_complex(
    "protein.pdb", "ligand.sdf")
print(result.vector.shape, result.schema)
```

## Feature anatomy

`84 + 15 + 360 + 114 + 456 + 108 = 1137`. Tokens are `15 + 7 + 32 + 3 = 57`. The canonical defaults are epsilon 0.5 Å, Gauss lambda 1.0, fourth-order quadrature, 4→5/5→6/6→7 Å windows, float64 stable spectral algebra, and K=8 train-only K-means with seed 0, n_init 20, max_iter 300.

## Training a vocabulary

```python
vocab = PGLVocabulary.fit(training_tokens)  # training rows only
vocab.save("vocab.npz")
```

Test/CASF tokens must never be used to fit mean, standard deviation, centers, or tau.

## CLI

```bash
pglrep featurize --protein protein.pdb --ligand ligand.sdf --vocabulary vocab.npz --output feature.npy
pglrep inspect --feature feature.npy
```

See `docs/algorithm.md`, `docs/api.md`, and `docs/reproducibility.md`. Database structures and paper experiment datasets are not included.
