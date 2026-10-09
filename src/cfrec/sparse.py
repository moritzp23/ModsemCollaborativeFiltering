"""Sparsity patterns for sparse approximations of item-item models."""
import numpy as np
import scipy.sparse as sp


def limit_nnz_per_column(A: sp.csc_matrix, max_in_col: int) -> sp.csc_matrix:
    """Keep only the `max_in_col` entries of largest magnitude in each column."""
    A = A.tocsc(copy=True)
    for j in np.flatnonzero(np.diff(A.indptr) > max_in_col):
        col = A.data[A.indptr[j]:A.indptr[j + 1]]  # view
        col[np.argsort(-np.abs(col))[max_in_col:]] = 0.
    A.eliminate_zeros()
    return A


def sparsify(B: np.ndarray, threshold: float, max_in_col: int | None = None) -> sp.csc_matrix:
    """Pattern of B by magnitude: entries with |B_ij| > threshold, at most `max_in_col` per column.

    With threshold == 0 the `max_in_col` largest entries per column are kept.
    """
    max_in_col = max_in_col or B.shape[0]
    A = sp.csc_matrix(np.where(np.abs(B) > threshold, B, 0.).astype(np.float32))
    return limit_nnz_per_column(A, max_in_col)


def standardized_gram(XtX: np.ndarray, n_users: int, alpha: float):
    """(X - mu)^T (X - mu), rescaled by (n_users * var)^(alpha / 2) from both sides.

    alpha = 1: correlation matrix, alpha = 0: (n_users times the) covariance matrix.
    Only valid for binary X (then mu = diag(X^T X) / n_users). Returns (C, scaling, rescaling)
    with C = diag(scaling) (X - mu)^T (X - mu) diag(scaling) and rescaling = 1 / scaling.
    """
    mu = np.diag(XtX) / n_users
    C = XtX - np.outer(mu, mu * n_users)
    rescaling = np.power(np.diag(C), alpha / 2.0)
    scaling = (1.0 / rescaling).astype(XtX.dtype)
    return scaling[:, None] * C * scaling[None, :], scaling, rescaling


def correlation_pattern(XtX: np.ndarray, n_users: int, alpha: float, threshold: float, max_in_col: int,
                        keep_diagonal: bool = False) -> sp.csc_matrix:
    """Sparsity pattern from thresholding the (alpha-)standardized covariance matrix in absolute value."""
    C, _, _ = standardized_gram(XtX, n_users, alpha)
    if not keep_diagonal:
        np.fill_diagonal(C, 0.)
    A = sp.csc_matrix(np.where(np.abs(C) > threshold, C, 0.).astype(np.float32))
    return limit_nnz_per_column(A, max_in_col)


def restrict_to_pattern(B: np.ndarray, A: sp.spmatrix) -> sp.csr_matrix:
    """Sparse matrix with the entries of B on the nonzero pattern of A."""
    rows, cols = A.nonzero()
    return sp.csr_matrix((B[rows, cols], (rows, cols)), shape=B.shape, dtype=np.float32)
