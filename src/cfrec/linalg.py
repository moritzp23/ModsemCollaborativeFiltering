import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky
from scipy.linalg.lapack import dpotri, spotri


def gram(X: sp.csr_matrix) -> np.ndarray:
    """Dense item-item Gram matrix X^T X."""
    return (X.T @ X).toarray()


def spd_inverse(A: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric positive definite matrix via Cholesky + LAPACK ?potri.

    Roughly twice as fast as np.linalg.inv; keeps the dtype (float32 or float64) of `A`.
    """
    potri = {np.dtype(np.float32): spotri, np.dtype(np.float64): dpotri}.get(A.dtype)
    if potri is None:
        raise ValueError(f'Unsupported dtype: {A.dtype}')
    inv, info = potri(cholesky(A), overwrite_c=True)
    if info != 0:
        raise np.linalg.LinAlgError(f'?potri failed with info={info}')
    # potri only fills the upper triangle
    inv += np.triu(inv, k=1).T
    return inv


def zero_diag_solution(P: np.ndarray) -> np.ndarray:
    """B = I - P diag(1 / diag(P)), i.e. the closed-form solution of a linear AE with diag(B) = 0."""
    B = P / (-np.diag(P))
    np.fill_diagonal(B, 0.)
    return B
