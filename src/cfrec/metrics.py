"""Vectorized ranking metrics on top-k index arrays.

Recall@k = #hits in top-k / min(k, #relevant)     (as in Liang et al. 2018)
NDCG@k   = DCG@k / IDCG@k with IDCG computed for min(k, #relevant) relevant items.
"""
import numpy as np
import scipy.sparse as sp
from bottleneck import argpartition


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores per row, sorted in descending order."""
    k = min(k, scores.shape[1])
    rows = np.arange(scores.shape[0])[:, None]
    idx = argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-scores[rows, idx], axis=1)
    return idx[rows, order]


def hits(topk_idx: np.ndarray, X_true: sp.csr_matrix) -> np.ndarray:
    """Boolean (n_users, k) array: is the j-th recommended item relevant for the user?"""
    rows = np.repeat(np.arange(topk_idx.shape[0]), topk_idx.shape[1])
    return np.asarray(X_true[rows, topk_idx.ravel()]).reshape(topk_idx.shape) > 0


def recall(hit: np.ndarray, n_true: np.ndarray, k: int) -> np.ndarray:
    return hit[:, :k].sum(axis=1) / np.minimum(k, n_true)


def ndcg(hit: np.ndarray, n_true: np.ndarray, k: int) -> np.ndarray:
    discount = 1. / np.log2(np.arange(2, k + 2))
    hit = hit[:, :k]
    dcg = (hit * discount[:hit.shape[1]]).sum(axis=1)
    idcg = np.cumsum(discount)[np.minimum(k, n_true) - 1]
    return dcg / idcg


METRICS = {'recall': recall, 'ndcg': ndcg}


def parse_metric(name: str):
    """'recall@20' -> (recall, 20)"""
    metric, k = name.lower().split('@')
    return METRICS[metric], int(k)
