"""Sparse approximation of the MRF / linear autoencoder solution (Steck 2019, "Markov Random Fields for
Collaborative Filtering", Section 3).

Items are nodes of a sparse graph obtained by thresholding the standardized covariance matrix. Instead of
inverting the full (n_items x n_items) matrix, only small blocks (item i and its neighbours) are inverted.
"""
import time

import numpy as np
import scipy.sparse as sp

from ..linalg import gram, spd_inverse
from ..sparse import limit_nnz_per_column, standardized_gram
from .base import ItemItemModel


class MRFApprox(ItemItemModel):
    """
    lmbda:      L2 regularization added to the diagonal of the standardized covariance matrix
    alpha:      standardization exponent (1: correlation, 0: covariance)
    threshold:  entries of the standardized covariance with |C_ij| <= threshold are dropped from the graph
    max_in_col: maximum number of neighbours per item
    r:          fraction of each block whose columns are taken from that block (0: only the center item)
    """

    def __init__(self, lmbda: float, alpha: float, threshold: float, max_in_col: int = 1000, r: float = 0.5,
                 verbose: bool = False):
        self.lmbda = lmbda
        self.alpha = alpha
        self.threshold = threshold
        self.max_in_col = max_in_col
        self.r = r
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def sparsity_pattern(self, C: np.ndarray) -> sp.csc_matrix:
        # the diagonal is kept, so that every block contains its center item
        A = sp.csc_matrix(np.where(np.abs(C) > self.threshold, C, 0.).astype(np.float32))
        return limit_nnz_per_column(A, self.max_in_col)

    def _blocks(self, A: sp.csc_matrix, diag: np.ndarray):
        """Steps 1, 2, 4 of Section 3.2: greedily cover all items with blocks (item + neighbours)."""
        col_count = np.diff(A.indptr)
        # process items with many neighbours first, ties broken by popularity
        order = np.argsort(col_count + diag / 2.0 / np.max(diag))[::-1]
        todo = np.ones(A.shape[0], dtype=bool)
        blocks = []
        for i in order:
            if todo[i]:
                rows = A.indices[A.indptr[i]:A.indptr[i + 1]]
                vals = A.data[A.indptr[i]:A.indptr[i + 1]]
                n_i = rows[np.argsort(np.abs(vals))[::-1]]
                if len(n_i) == 0:  # isolated item without any entry above the threshold (not even its diagonal)
                    continue
                blocks.append(n_i)
                todo[n_i[:max(1, int(np.ceil(len(n_i) * self.r)))]] = False
        return blocks

    def fit(self, X: sp.csr_matrix):
        start = time.perf_counter()
        n_items = X.shape[1]
        XtX = gram(X)
        self.mu = np.diag(XtX) / X.shape[0]
        C, scaling, rescaling = standardized_gram(XtX, X.shape[0], self.alpha)
        del XtX
        A = self.sparsity_pattern(C)
        C[np.diag_indices_from(C)] += self.lmbda
        self._log(f'pattern: density {A.nnz / n_items ** 2:.5f}')

        blocks = self._blocks(A, np.diag(C))
        self._log(f'{len(blocks)} blocks')

        # step 3: dense solution on each block, keep the columns of the first ceil(r * |block|) items
        rows, cols, vals = [], [], []
        for n_i in blocks:
            B_block = spd_inverse(C[np.ix_(n_i, n_i)])
            B_block /= -np.diag(B_block)
            d = n_i[:max(1, int(np.ceil(len(n_i) * self.r)))]
            rr, cc = np.meshgrid(n_i, d, indexing='ij')
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(B_block[:, :len(d)].ravel())
        del C

        # average the solutions for entries covered by multiple blocks
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        shape = (n_items, n_items)
        B_sum = sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
        B_cnt = sp.coo_matrix((np.ones_like(vals), (rows, cols)), shape=shape).tocsr()
        B = B_sum.copy()
        B.data = B_sum.data / B_cnt.data  # same sparsity structure, both summed over duplicates
        B.setdiag(0.)
        B.eliminate_zeros()

        # force the pattern of A onto B and undo the standardization
        B = B.multiply(A.tocsr() != 0).tocsr()
        self.B = (sp.diags(scaling) @ B @ sp.diags(rescaling.astype(np.float32))).astype(np.float32).tocsr()
        self.fit_time = time.perf_counter() - start
        self._log(f'fit: {self.fit_time:.1f}s, density of B: {self.density:.5f}')
        return self
