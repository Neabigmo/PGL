import numpy as np
from pglrep import PGLVocabulary

training_tokens = np.load("training_tokens.npy")
PGLVocabulary.fit(training_tokens).save("train_vocab.npz")
