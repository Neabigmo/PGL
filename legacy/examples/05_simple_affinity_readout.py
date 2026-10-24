import numpy as np
from sklearn.linear_model import Ridge

# X must be built from training-only PGL vectors; y are training labels.
X, y = np.load("training_vectors.npy"), np.load("training_labels.npy")
model = Ridge(alpha=1.0).fit(X, y)
print(model.predict(X[:1]))
