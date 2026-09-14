from pathlib import Path
from pglrep import PGLRep, PGLVocabulary

model = PGLRep(vocabulary=PGLVocabulary.load("train_vocab.npz"))
for case in Path("cases").iterdir():
    if case.is_dir():
        result = model.from_complex(case / "protein.pdb", case / "ligand.sdf")
        print(case.name, result.vector.shape)
