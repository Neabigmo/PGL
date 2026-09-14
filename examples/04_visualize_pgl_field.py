from pglrep import PGLRep, PGLVocabulary

result = PGLRep(vocabulary=PGLVocabulary.load("train_vocab.npz")).from_complex("protein.pdb", "ligand.sdf")
print("central token matrix:", result.edge_tokens.shape)
