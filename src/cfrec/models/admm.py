"""Sparse linear autoencoder (SLIM-type) trained with ADMM (Steck et al., WSDM 2020,
"ADMM SLIM: Sparse Recommendations for Many Users"), with the EDLAE regularizer:

    min_B ||X - X B||_F^2 + lambda1 ||B||_1 + ||Lambda^{1/2} B||_F^2   s.t. diag(B) = 0,
    Lambda = p / (1 - p) * diagMat(diag(X^T X)) + lambda2 * I.

p = 0 gives the original ADMM SLIM objective.

Note: the seminar version (legacy/Models.py) passed `B + Gamma / rho` into a soft-threshold function that
added `Gamma / rho` again, i.e. it thresholded `B + Gamma / rho + lambda1 / rho^2`. This implementation
uses the standard update C = S_{lambda1 / rho}(B + Gamma / rho); results can differ slightly from the
report for the tuned lambda1.
"""
import time

import numpy as np
import scipy.sparse as sp

from ..linalg import gram, spd_inverse
from .base import ItemItemModel


class ADMMSlim(ItemItemModel):

    def __init__(self, lambda1: float, lambda2: float, p: float = 0., rho: float = 10_000., n_iter: int = 50,
                 nonneg: bool = False, check_convergence: bool = False, adjust_rho: bool = False,
                 eps_abs: float = 1e-3, eps_rel: float = 1e-3, t: float = 10., tau: float = 2., verbose: bool = False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.p = p
        self.rho = rho
        self.n_iter = n_iter
        self.nonneg = nonneg
        self.check_convergence = check_convergence
        self.adjust_rho = adjust_rho
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.t = t
        self.tau = tau
        self.verbose = verbose

    def _soft_threshold(self, x):
        thr = self.lambda1 / self.rho
        if self.nonneg:
            return np.maximum(x - thr, 0.)
        return np.sign(x) * np.maximum(np.abs(x) - thr, 0.)

    def _residuals(self, B, C, C_old, Gamma):
        n = B.shape[0]
        eps_primal = self.eps_abs * n + self.eps_rel * max(np.linalg.norm(B), np.linalg.norm(C))
        eps_dual = self.eps_abs * n + self.eps_rel * np.linalg.norm(Gamma)
        r_primal = np.linalg.norm(B - C)
        r_dual = np.linalg.norm(C - C_old) * self.rho
        return r_primal, r_dual, eps_primal, eps_dual

    def fit(self, X: sp.csr_matrix):
        start = time.perf_counter()
        G = gram(X)
        diag = self.p / (1 - self.p) * np.diag(G) + self.lambda2 + self.rho
        G[np.diag_indices_from(G)] += diag
        P = spd_inverse(G)
        del G
        B_aux = -P * diag[None, :]
        B_aux[np.diag_indices_from(B_aux)] += 1.
        diag_P = np.diag(P).copy()

        Gamma = np.zeros_like(P)
        C = np.zeros_like(P)
        self.history = []
        for it in range(self.n_iter):
            C_old = C
            B_tilde = B_aux + P @ (self.rho * C - Gamma)
            B = B_tilde - P * (np.diag(B_tilde) / diag_P)[None, :]  # enforce diag(B) = 0
            C = self._soft_threshold(B + Gamma / self.rho)
            Gamma += self.rho * (B - C)

            if self.check_convergence:
                r_p, r_d, eps_p, eps_d = self._residuals(B, C, C_old, Gamma)
                self.history.append(dict(iter=it, density=np.count_nonzero(C) / C.size, r_primal=r_p, r_dual=r_d))
                if self.verbose:
                    print(f'iter {it}: density {self.history[-1]["density"]:.5f}, '
                          f'primal {r_p:.3g} (eps {eps_p:.3g}), dual {r_d:.3g} (eps {eps_d:.3g})')
                if r_p < eps_p and r_d < eps_d:
                    break
                if self.adjust_rho:  # note: P is not recomputed, as in the seminar version
                    if r_p > self.t * r_d:
                        self.rho *= self.tau
                    elif r_d > self.t * r_p:
                        self.rho /= self.tau
            elif self.verbose:
                print(f'iter {it}: density {np.count_nonzero(C) / C.size:.5f}')

        self.B = sp.csr_matrix(C)
        self.fit_time = time.perf_counter() - start
        return self
