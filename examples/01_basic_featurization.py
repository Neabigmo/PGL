from pglrep import PGLConfig, PGLRep, PGLVocabulary

vocab = PGLVocabulary.load("train_vocab.npz")
result = PGLRep(PGLConfig(), vocab).from_complex("protein.pdb", "ligand.sdf")
print(result.vector.shape, result.schema)
