from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .metrics import hits, parse_metric, topk

DEFAULT_METRICS = ('recall@20', 'recall@50', 'ndcg@100')


def recommend(model, X: sp.csr_matrix, k: int, batch_size: int = 5000) -> np.ndarray:
    """Top-k items per user, excluding the items the user already interacted with."""
    out = []
    for start in range(0, X.shape[0], batch_size):
        X_batch = X[start:start + batch_size]
        scores = np.asarray(model.score(X_batch), dtype=np.float32)
        rows, cols = X_batch.nonzero()
        scores[rows, cols] = -np.inf
        out.append(topk(scores, k))
    return np.vstack(out)


def evaluate(model, X_in: sp.csr_matrix, X_out: sp.csr_matrix, metrics=DEFAULT_METRICS,
             batch_size: int = 5000, per_user: bool = False) -> dict:
    """Strong-generalization evaluation: fold in `X_in`, rank against held-out `X_out`.

    Users without held-out items are skipped. Returns {metric: mean, metric_se: standard error};
    with `per_user=True` the per-user values are returned as well (key `per_user`).
    """
    n_true = np.diff(X_out.indptr)
    keep = n_true > 0
    X_in, X_out, n_true = X_in[keep], X_out[keep], n_true[keep]

    parsed = {m: parse_metric(m) for m in metrics}
    max_k = max(k for _, k in parsed.values())
    hit = hits(recommend(model, X_in, max_k, batch_size), X_out)

    values = {m: fn(hit, n_true, k) for m, (fn, k) in parsed.items()}
    result = {}
    for m, v in values.items():
        result[m] = float(v.mean())
        result[f'{m}_se'] = float(v.std() / np.sqrt(len(v)))
    if per_user:
        result['per_user'] = values
    return result


def format_result(result: dict) -> str:
    return ', '.join(f'{m}: {v:.4f} ± {result[m + "_se"]:.4f}'
                     for m, v in result.items() if not m.endswith('_se') and m != 'per_user')
