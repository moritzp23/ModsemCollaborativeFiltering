import numpy as np
import scipy.sparse as sp
from bottleneck import argpartition

from ..linalg import gram
from .base import Recommender


class ItemKNN(Recommender):
    """Item-to-item collaborative filtering with a truncated (k nearest neighbours) similarity matrix.

    similarity: 'cosine' or 'pearson'. `alpha` interpolates the normalization: the Gram matrix is divided
    by norm_i^alpha * norm_j^alpha (alpha=1: cosine / Pearson correlation).
    """

    def __init__(self, similarity='cosine', num_neighbors=100, alpha=1.0, renormalize_similarity=False,
                 renormalization_interval='[0,1]', enable_average_bias=False, min_similarity_threshold=0.,
                 trunc_entries=False, trunc_val=0., l1_normalization=False, eps=1e-9):
        self.similarity = similarity
        self.num_neighbors = num_neighbors
        self.alpha = alpha
        self.renormalize_similarity = renormalize_similarity
        self.renormalization_interval = renormalization_interval
        self.enable_average_bias = enable_average_bias
        self.min_similarity_threshold = min_similarity_threshold
        self.trunc_entries = trunc_entries
        self.trunc_val = trunc_val
        self.l1_normalization = l1_normalization
        self.eps = eps

    def _similarity(self, X):
        S = gram(X)
        if self.similarity == 'pearson':
            # Pearson correlation = cosine similarity of the mean-centered columns (may suffer from cancellation)
            mu = np.asarray(X.mean(axis=0)).ravel()
            S -= X.shape[0] * np.outer(mu, mu)
        elif self.similarity != 'cosine':
            raise NotImplementedError(f'similarity={self.similarity} is not supported.')
        inv_norm = 1 / np.maximum(np.power(np.diag(S), self.alpha / 2.0), self.eps)
        S = S * inv_norm[None, :] * inv_norm[:, None]
        np.fill_diagonal(S, 0.)
        return S

    def fit(self, X: sp.csr_matrix):
        n_items = X.shape[1]
        S = self._similarity(X)

        if self.renormalize_similarity:
            lo, hi = S.min(), S.max()
            if self.renormalization_interval == '[0,1]':
                S = (S - lo) / (hi - lo)
            elif self.renormalization_interval == '[-1,1]':
                S = 2 * (S - lo) / (hi - lo) - 1
            else:
                raise ValueError(f'{self.renormalization_interval} is not a supported interval')
            np.fill_diagonal(S, 0.)

        if self.trunc_entries:
            S[S < self.min_similarity_threshold] = self.trunc_val

        # keep the num_neighbors largest (in absolute value) similarities per row
        rows = np.arange(n_items)[:, None]
        idx = argpartition(-np.abs(S), self.num_neighbors - 1, axis=1)[:, :self.num_neighbors]
        S_knn = np.zeros(S.shape, dtype=np.float32)
        S_knn[rows, idx] = S[rows, idx]

        if self.l1_normalization:
            S_knn /= np.maximum(np.abs(S_knn).sum(axis=1), self.eps)[:, None]

        self.sim = sp.csr_matrix(S_knn)
        self.item_mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, X: sp.csr_matrix) -> np.ndarray:
        scores = (X @ self.sim.T).toarray()
        if self.enable_average_bias:
            # prediction on mean-centered data: (X - mu) S^T + mu
            scores -= self.sim @ self.item_mean - self.item_mean
        return scores
