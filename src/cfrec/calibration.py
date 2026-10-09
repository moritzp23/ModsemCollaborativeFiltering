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
