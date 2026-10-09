"""Linear autoencoders with closed-form solutions (Steck 2019, Steck 2020)."""
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve

from ..linalg import gram, spd_inverse, zero_diag_solution
from .base import ItemItemModel, Recommender


class EDLAE(ItemItemModel):
    """Emphasized Denoising Linear Autoencoder (Steck, NeurIPS 2020), with constraint diag(B) = 0:

        min_B ||X - X B||_F^2 + ||Lambda^{1/2} B||_F^2   s.t. diag(B) = 0,
        Lambda = p / (1 - p) * diagMat(diag(X^T X)) + lambda * I.

    Closed form: B = I - P diagMat(1 / diag(P)), P = (X^T X + Lambda)^{-1}.
    p = 0 recovers EASE (Steck, WWW 2019).
    """

    def __init__(self, lmbda: float, p: float = 0.):
        self.lmbda = lmbda
        self.p = p

    def regularized_gram(self, X: sp.csr_matrix) -> np.ndarray:
        """X^T X + Lambda."""
        G = gram(X)
        G[np.diag_indices_from(G)] *= 1 + self.p / (1 - self.p)
        G[np.diag_indices_from(G)] += self.lmbda
        return G

    def fit(self, X: sp.csr_matrix):
        start = time.perf_counter()
        self.B = zero_diag_solution(spd_inverse(self.regularized_gram(X)))
        self.fit_time = time.perf_counter() - start
        return self


class EASE(EDLAE):
    """Embarrassingly Shallow Autoencoder (Steck, WWW 2019): EDLAE with p = 0."""

    def __init__(self, lmbda: float):
        super().__init__(lmbda=lmbda, p=0.)


class DLAE(Recommender):
    """Denoising Linear Autoencoder (Steck, NeurIPS 2020), no diagonal constraint:

        B = (X^T X + Lambda)^{-1} X^T X = I - (X^T X + Lambda)^{-1} Lambda.

    method='chol' never forms B; predictions solve a linear system with the Cholesky factor:
        X B = X - X (X^T X + Lambda)^{-1} Lambda.
    method='inv' computes B explicitly.
    """

    def __init__(self, lmbda: float, p: float, method: str = 'chol'):
        if method not in ('chol', 'inv'):
            raise NotImplementedError(f'{method} is not a supported method.')
        self.lmbda = lmbda
        self.p = p
        self.method = method

    def fit(self, X: sp.csr_matrix):
        start = time.perf_counter()
        G = gram(X)
        self.Lambda = self.p / (1 - self.p) * np.diag(G) + self.lmbda
        G[np.diag_indices_from(G)] += self.Lambda
        if self.method == 'chol':
            self.cho = cho_factor(G, lower=True, overwrite_a=True)
        else:
            self.B = np.eye(G.shape[0], dtype=G.dtype) - spd_inverse(G) * self.Lambda[None, :]
        self.fit_time = time.perf_counter() - start
        return self

    def score(self, X: sp.csr_matrix) -> np.ndarray:
        X_dense = X.toarray()
        if self.method == 'inv':
            return X_dense @ self.B
        return X_dense - cho_solve(self.cho, X_dense.T).T * self.Lambda[None, :]


class MRFDense(ItemItemModel):
    """Dense Markov random field (Steck 2019, "Markov Random Fields for Collaborative Filtering"):
    B from the precision matrix of the (regularized) empirical covariance, with diag(B) = 0.

    With `mean_removal`, predictions are mu + (X - mu) B.
    """

    def __init__(self, lmbda: float, mean_removal: bool = False):
        self.lmbda = lmbda
        self.mean_removal = mean_removal

    def fit(self, X: sp.csr_matrix):
        start = time.perf_counter()
        n_users = X.shape[0]
        C = gram(X)
        self.mu = np.diag(C) / n_users  # only valid for binary data
        C -= np.outer(self.mu, self.mu * n_users)  # n_users * covariance
        C[np.diag_indices_from(C)] += self.lmbda
        self.B = zero_diag_solution(spd_inverse(C))
        self.fit_time = time.perf_counter() - start
        return self

    def score(self, X: sp.csr_matrix) -> np.ndarray:
        if self.mean_removal:
            return self.mu + (X.toarray() - self.mu) @ self.B
        return X @ self.B
