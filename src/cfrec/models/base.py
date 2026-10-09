from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Recommender:
    """Interface: `fit` on a binary user x item csr matrix, `score` a batch of (fold-in) users.

    Masking of already-seen items and top-k selection are handled by `cfrec.evaluation`.
    """

    def fit(self, X: sp.csr_matrix) -> 'Recommender':
        raise NotImplementedError

    def score(self, X: sp.csr_matrix) -> np.ndarray:
        raise NotImplementedError


class ItemItemModel(Recommender):
    """Models predicting scores = X @ B with an item x item matrix B (dense ndarray or sparse)."""

    B: np.ndarray | sp.spmatrix

    def score(self, X: sp.csr_matrix) -> np.ndarray:
        if sp.issparse(self.B):
            return (X @ self.B).toarray()
        return X @ self.B

    @property
    def density(self) -> float:
        nnz = self.B.nnz if sp.issparse(self.B) else np.count_nonzero(self.B)
        return nnz / (self.B.shape[0] * self.B.shape[1])
