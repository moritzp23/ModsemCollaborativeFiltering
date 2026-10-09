import numpy as np
import scipy.sparse as sp

from .base import Recommender


class MostPopular(Recommender):
    """Recommends the globally most popular items (not yet seen by the user)."""

    def fit(self, X: sp.csr_matrix):
        self.item_counts = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, X: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.item_counts, (X.shape[0], 1))
