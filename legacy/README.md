# Legacy: earlier internal readout

This directory preserves an earlier internal implementation of a
1,137-dimensional PGL readout. It is kept for reference and for reproducing
internal analyses; the paper and the released package do not use it.

The legacy path ships chemistry blocks, a K-means token vocabulary, VLAD
pooling and a spectral response block, and needs PyTorch. It is not installed
with the package, has no command line entry point and is not covered by the
test suite.

Layout:

- `pglrep/`: the legacy package (import name `pglrep`).
- `examples/`: scripts written for the legacy package.
- `tests/`: tests for the legacy package. They require PyTorch and are not
  collected by the repository test run.
- `docs/`: the API and reproducibility notes that belonged to the legacy
  package.

To use it, put this directory on the import path:

```bash
PYTHONPATH=legacy python -c "import pglrep; print(pglrep.__version__)"
```

The paper's representation is the 287-dimensional contact-token block in
`src/pgl/`; see `docs/algorithm.md` and `docs/features.md`.
