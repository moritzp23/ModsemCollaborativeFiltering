"""Low-rank approximations B ~ U V^T of a fitted dense item-item model (report Section 3.3.1).

Predictions are computed as (X U) V^T, which costs O(n_items * rank) per user instead of O(n_items^2).
"""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

from .base import Recommender


class LowRank(Recommender):
    def __init__(self, U: np.ndarray, V: np.ndarray):
        self.U = U
        self.V = V

    def fit(self, X: sp.csr_matrix):
        return self  # constructed from a fitted model, see the factories below

    def score(self, X: sp.csr_matrix) -> np.ndarray:
        return (X @ self.U) @ self.V.T

    @property
    def rank(self) -> int:
        return self.U.shape[1]


class LowRankFactorization:
    """Precomputed decomposition of B; `truncate(k)` returns the rank-k model.

    method='svd':  B ~ U_k S_k V_k^T (best rank-k approximation of B in Frobenius norm).
    method='eig':  projection B V_k V_k^T onto the top-k eigenvectors of B^T G B with G = X^T X + Lambda,
                   i.e. the rank-k approximation minimizing the training objective ||X - X B'||^2 + ||Lambda^{1/2} B'||^2
                   among projections of B (see the report). Requires the regularized Gram matrix `G`.
    """

    def __init__(self, B: np.ndarray, method: str = 'svd', G: np.ndarray | None = None):
        self.method = method
        if method == 'svd':
            U, S, Vt = np.linalg.svd(B)
            self.U, self.V = U * S[None, :], Vt.T  # sorted by decreasing singular value
        elif method == 'eig':
            if G is None:
                raise ValueError("method='eig' requires the regularized Gram matrix G")
            _, V = eigh(B.T @ G @ B, driver='evd')
            self.V = V[:, ::-1]  # decreasing eigenvalues
            self.U = B @ self.V
        else:
            raise NotImplementedError(method)

    def truncate(self, k: int) -> LowRank:
        return LowRank(np.ascontiguousarray(self.U[:, :k]), np.ascontiguousarray(self.V[:, :k]))
