# API

`PGLConfig` exposes the canonical numerical settings. `PGLVocabulary.fit`, `.save`, and `.load` manage the train-only K=8 codebook. `PGLRep(...).from_complex(protein_path, ligand_path)` returns `PGLResult` with `vector`, `schema`, `metadata`, central `edge_tokens`, spectra, and Gauss diagnostics.
