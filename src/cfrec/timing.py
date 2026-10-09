"""Inference latency benchmarks (report Tables 3.3 / 3.4 / A.1).

A predictor is a pair (prepare, predict): `prepare` converts a csr batch into the model's input format
(not timed), `predict` computes the top-k item indices (timed). Following the report, we measure
  - batch time: top-k for a batch of 1000 random users,
  - query time: top-k for a single user, averaged over 100 random users,
each as the minimum over 7 repeats of `timeit` with automatic number of loops.
"""
import timeit

import numpy as np
import scipy.sparse as sp
from bottleneck import argpartition
from sparse_dot_topn import sp_matmul_topn


def _topk_masked(pred, x_dense, k):
    pred[x_dense != 0] = -np.inf
    return argpartition(-pred, k - 1, axis=1)[:, :k]


def dense_predictor(B: np.ndarray, k: int = 100):
    return (lambda X: X.toarray(),
            lambda x: _topk_masked(x @ B, x, k))


def lowrank_predictor(U: np.ndarray, V: np.ndarray, k: int = 100):
    Vt = np.ascontiguousarray(V.T)
    return (lambda X: X.toarray(),
            lambda x: _topk_masked((x @ U) @ Vt, x, k))


def popularity_predictor(item_counts: np.ndarray, k: int = 100):
    return (lambda X: X.toarray(),
            lambda x: _topk_masked(np.tile(item_counts, (x.shape[0], 1)), x, k))


def sparse_predictor(B: sp.spmatrix, k: int = 100, n_threads: int = 16):
    """Sparse x sparse product with top-n selection; -inf on the diagonal removes already seen items."""
    B = (sp.csr_matrix(B, dtype=np.float32) - np.inf * sp.identity(B.shape[0], dtype=np.float32, format='csr')).tocsr()

    def predict(x):
        threads = n_threads if x.shape[0] > 1 else None
        return sp_matmul_topn(x, B, top_n=k, n_threads=threads)
    return (lambda X: sp.csr_matrix(X, dtype=np.float32)), predict


def model_predictor(model, k: int = 100):
    """Generic fallback using `model.score`."""
    return (lambda X: X,
            lambda x: _topk_masked(np.asarray(model.score(x)), x.toarray(), k))


def _best_time(fn, repeat=7):
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat=repeat, number=number)) / number


def latency(predictor, X: sp.csr_matrix, n_batch: int = 1000, n_queries: int = 100, seed: int = 42,
            repeat: int = 7) -> dict:
    prepare, predict = predictor
    rng = np.random.default_rng(seed)
    batch = prepare(X[rng.integers(0, X.shape[0], size=n_batch)])
    batch_time = _best_time(lambda: predict(batch), repeat)
    queries = [prepare(X[[i]]) for i in rng.integers(0, X.shape[0], size=n_queries)]
    query_time = np.mean([_best_time(lambda q=q: predict(q), repeat) for q in queries])
    return {'batch_time': batch_time, 'query_time': float(query_time)}
