"""Calibration of predicted interaction probabilities P(x_i = 1 | y) and of the number of hidden items.

All quantities are computed on items the user has *not* interacted with in the fold-in set y; the label of
such an item is 1 iff it is in the user's held-out set.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

from .metrics import topk

# probability bins, finer near 0 where most of the mass is
BIN_EDGES = np.array([0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 + 1e-9])


def _batches(n, batch_size):
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def fit_platt(score_fn: Callable[[sp.csr_matrix], np.ndarray], X_in: sp.csr_matrix, X_out: sp.csr_matrix,
              n_iter: int = 25, batch_size: int = 1000) -> tuple[float, float]:
    """Platt scaling p = sigmoid(a * s + b), fitted by Newton's method on all unobserved (user, item) pairs."""
    w = np.zeros(2)
    scores, labels = [], []
    for sl in _batches(X_in.shape[0], batch_size):
        s = np.asarray(score_fn(X_in[sl]), dtype=np.float64)
        unobs = X_in[sl].toarray() == 0
        scores.append(s[unobs])
        labels.append(X_out[sl].toarray()[unobs])
    s, y = np.concatenate(scores), np.concatenate(labels)
    s_mean, s_std = s.mean(), s.std()
    z = (s - s_mean) / s_std
    for _ in range(n_iter):
        p = expit(w[0] * z + w[1])
        g = np.array([np.dot(p - y, z), np.sum(p - y)])
        r = p * (1 - p)
        H = np.array([[np.dot(r, z * z), np.dot(r, z)], [np.dot(r, z), r.sum()]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    a = w[0] / s_std
    return float(a), float(w[1] - a * s_mean)


class CalibrationAccumulator:
    """Streams batches of (probabilities, fold-in, held-out) and accumulates calibration statistics."""

    def __init__(self, top_k: int = 100):
        self.top_k = top_k
        n_bins = len(BIN_EDGES) - 1
        self.bins = {scope: np.zeros((3, n_bins)) for scope in ('all', f'top{top_k}')}  # sum_p, sum_y, count
        self.sq_err = {scope: 0. for scope in self.bins}
        self.nll = {scope: 0. for scope in self.bins}
        self.n_users = 0
        self.counts = {'true': [], 'mean': [], 'var': []}

    def _add(self, scope, p, y):
        idx = np.digitize(p, BIN_EDGES) - 1
        n_bins = len(BIN_EDGES) - 1
        self.bins[scope] += np.stack([np.bincount(idx, p, n_bins), np.bincount(idx, y, n_bins),
                                      np.bincount(idx, minlength=n_bins)])
        self.sq_err[scope] += np.sum((p - y) ** 2)
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        self.nll[scope] -= np.sum(y * np.log(pc) + (1 - y) * np.log(1 - pc))

    def update(self, probs: np.ndarray, X_in: sp.csr_matrix, X_out: sp.csr_matrix):
        probs = np.asarray(probs, dtype=np.float64)
        unobs = X_in.toarray() == 0
        labels = X_out.toarray()
        self._add('all', probs[unobs], labels[unobs])

        masked = np.where(unobs, probs, -np.inf)
        top = topk(masked, self.top_k)
        rows = np.arange(len(top))[:, None]
        self._add(f'top{self.top_k}', probs[rows, top].ravel(), labels[rows, top].ravel())

        p_unobs = np.where(unobs, probs, 0.)
        self.counts['true'].append(labels.sum(axis=1))
        self.counts['mean'].append(p_unobs.sum(axis=1))
        self.counts['var'].append((p_unobs * (1 - p_unobs)).sum(axis=1))
        self.n_users += len(probs)

    def reliability(self, scope='all'):
        sum_p, sum_y, cnt = self.bins[scope]
        keep = cnt > 0
        return {'mean_pred': sum_p[keep] / cnt[keep], 'freq': sum_y[keep] / cnt[keep], 'count': cnt[keep]}

    def summary(self) -> dict:
        out = {}
        for scope, (sum_p, sum_y, cnt) in self.bins.items():
            total = cnt.sum()
            out[f'ece_{scope}'] = float(np.sum(np.abs(sum_p - sum_y)) / total)
            out[f'brier_{scope}'] = float(self.sq_err[scope] / total)
            out[f'nll_{scope}'] = float(self.nll[scope] / total)
            out[f'mean_pred_{scope}'] = float(sum_p.sum() / total)
            out[f'base_rate_{scope}'] = float(sum_y.sum() / total)
        true, mean, var = (np.concatenate(self.counts[k]) for k in ('true', 'mean', 'var'))
        out['count_mae'] = float(np.mean(np.abs(mean - true)))
        out['count_bias'] = float(np.mean(mean - true))
        # central intervals under independent Bernoulli marginals (normal approximation of the Poisson-binomial)
        sd = np.sqrt(var)
        for level, zq in ((50, 0.6745), (90, 1.6449)):
            out[f'count_cover{level}_indep'] = float(np.mean(np.abs(true - mean) <= zq * sd + 0.5))
        return out


def sample_count_coverage(sample_fn: Callable, X_in: sp.csr_matrix, X_out: sp.csr_matrix, n_samples: int,
                          batch_size: int = 500) -> dict:
    """Coverage of central posterior intervals for the number of hidden items, from joint posterior samples.

    `sample_fn(X_batch, n_samples)` returns samples of shape (n_samples, batch, n_items); a sampled item counts
    as present if its value exceeds 0.5.
    """
    true, samples = [], []
    for sl in _batches(X_in.shape[0], batch_size):
        unobs = X_in[sl].toarray() == 0
        s = np.asarray(sample_fn(X_in[sl], n_samples))
        samples.append(((s > 0.5) & unobs[None]).sum(axis=2).T)  # (batch, n_samples)
        true.append(X_out[sl].toarray().sum(axis=1))
    true, counts = np.concatenate(true), np.concatenate(samples)
    out = {'count_mae_samples': float(np.mean(np.abs(counts.mean(axis=1) - true))),
           'count_bias_samples': float(np.mean(counts.mean(axis=1) - true))}
    for level in (50, 90):
        lo, hi = np.percentile(counts, [50 - level / 2, 50 + level / 2], axis=1)
        out[f'count_cover{level}_samples'] = float(np.mean((true >= lo) & (true <= hi)))
    return out


# --------------------------------------------------------------------------------------------------
# joint uncertainty: posterior samples vs. independent Bernoulli draws with the same marginals
# --------------------------------------------------------------------------------------------------

def _crps_samples(samples: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS of scalar predictive samples (users, m) against observations (users,): E|X - y| - E|X - X'| / 2."""
    m = samples.shape[1]
    s = np.sort(samples, axis=1).astype(np.float64)
    term1 = np.abs(s - obs[:, None]).mean(axis=1)
    weights = (2 * np.arange(1, m + 1) - m - 1) / m ** 2       # E|X - X'| = (2/m^2) sum_i (2i - m - 1) x_(i)
    return term1 - (s * weights).sum(axis=1)


def _energy_score(B: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Energy score of binary sample vectors B (m, users, d) against x (users, d), Euclidean norm:
    E||B - x|| - E||B - B'|| / 2 (pairs of consecutive samples for the second term)."""
    d_obs = np.sqrt((B != x[None]).sum(axis=2)).mean(axis=0)
    d_pair = np.sqrt((B[:-1] != B[1:]).sum(axis=2)).mean(axis=0)
    return d_obs - d_pair / 2


class JointAccumulator:
    """Joint-uncertainty scores of binary posterior samples on the unobserved items.

    Per user: the number of hidden items N and the number of hits H among the top-`top_k` items (ranked by
    the marginal probabilities), each scored by CRPS and central-interval coverage; and the energy score
    of the whole hidden-item vector. Lower CRPS / energy score is better.
    """

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.stats = {k: [] for k in ('crps_count', 'crps_hits', 'energy', 'cover50_count', 'cover90_count',
                                      'cover50_hits', 'cover90_hits')}

    def update(self, B: np.ndarray, probs: np.ndarray, X_in: sp.csr_matrix, X_out: sp.csr_matrix):
        """B: binary samples (m, users, items), already zero on observed items."""
        x = X_out.toarray() > 0
        unobs = X_in.toarray() == 0
        top = topk(np.where(unobs, probs, -np.inf), self.top_k)
        rows = np.arange(len(top))[:, None]
        for name, pred, obs in (('count', B.sum(axis=2).T, x.sum(axis=1)),
                                ('hits', B[:, rows, top].sum(axis=2).T, x[rows, top].sum(axis=1))):
            self.stats[f'crps_{name}'].append(_crps_samples(pred, obs))
            for level in (50, 90):
                lo, hi = np.percentile(pred, [50 - level / 2, 50 + level / 2], axis=1)
                self.stats[f'cover{level}_{name}'].append((obs >= lo) & (obs <= hi))
        self.stats['energy'].append(_energy_score(B, x & unobs))

    def summary(self) -> dict:
        return {k: float(np.concatenate(v).mean()) for k, v in self.stats.items()}


def sample_diagnostics(S: np.ndarray, probs: np.ndarray, unobs: np.ndarray) -> dict:
    """Are flow samples (m, users, items) near-binary, and does their mean match the one-step mean?"""
    vals = S[:, unobs]
    return {'frac_nonbinary': float(np.mean((vals > 0.1) & (vals < 0.9))),
            'frac_above_half': float(np.mean(vals > 0.5)),
            'mean_abs_gap_to_onestep': float(np.abs(S.mean(axis=0)[unobs] - probs[unobs]).mean())}
