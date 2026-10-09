"""Sparse and low-rank approximations of a fitted EDLAE (report Section 3.3, Tables 3.5 and A.1).

    python scripts/approximations.py ml-20m lowrank --method eig --ranks 100 1000 5000
    python scripts/approximations.py ml-20m magnitude          # |B_ij| > t  ("straightforward")
    python scripts/approximations.py ml-20m correlation        # pattern |C_alpha| > threshold
    python scripts/approximations.py ml-20m mrf                # MRF sparse approximation sweep

Writes a csv to results/ with ranking metrics and (with --timing) inference latencies.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from cfrec.configs import BEST_PARAMS, CORRELATION_THRESHOLDS, MAGNITUDE_THRESHOLDS, MRF_SWEEP
from cfrec.data import load_dataset
from cfrec.evaluation import evaluate, format_result
from cfrec.linalg import gram
from cfrec.models import EDLAE, ItemItemModel, LowRankFactorization, MRFApprox
from cfrec.sparse import correlation_pattern, restrict_to_pattern, sparsify
from cfrec.timing import latency, lowrank_predictor, sparse_predictor

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('dataset', choices=['ml-20m', 'netflix', 'msd'])
parser.add_argument('kind', choices=['lowrank', 'magnitude', 'correlation', 'mrf'])
parser.add_argument('--method', choices=['svd', 'eig'], default='eig', help='low-rank method')
parser.add_argument('--ranks', type=int, nargs='+', default=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
parser.add_argument('--timing', action='store_true', help='also measure inference latency')
parser.add_argument('--root', default='data')
args = parser.parse_args()

data = load_dataset(args.dataset, args.root)
print(data)


class Fixed(ItemItemModel):
    def __init__(self, B):
        self.B = B


def record(model, predictor=None, **info):
    res = evaluate(model, data.test_tr, data.test_te)
    if args.timing and predictor is not None:
        res.update(latency(predictor, data.test_tr))
    print(info, format_result(res))
    return {**info, **res}


rows = []
if args.kind == 'mrf':
    for cfg in MRF_SWEEP[args.dataset]:
        cfg = dict(cfg)
        target = cfg.pop('density')
        model = MRFApprox(max_in_col=1000, **cfg).fit(data.train)
        rows.append(record(model, sparse_predictor(model.B), target_density=target, density=model.density,
                           fit_time=model.fit_time, **cfg))
else:
    edlae = EDLAE(**BEST_PARAMS['edlae'][args.dataset]).fit(data.train)
    print(f'EDLAE fit: {edlae.fit_time:.1f}s')
    if args.kind == 'lowrank':
        start = time.perf_counter()
        G = edlae.regularized_gram(data.train) if args.method == 'eig' else None
        fac = LowRankFactorization(edlae.B, method=args.method, G=G)
        print(f'{args.method} decomposition: {time.perf_counter() - start:.1f}s')
        for k in args.ranks:
            model = fac.truncate(k)
            rows.append(record(model, lowrank_predictor(model.U, model.V), method=args.method, rank=k))
    elif args.kind == 'magnitude':
        for target, threshold in MAGNITUDE_THRESHOLDS[args.dataset].items():
            model = Fixed(restrict_to_pattern(edlae.B, sparsify(edlae.B, threshold)))
            rows.append(record(model, sparse_predictor(model.B), target_density=target, density=model.density,
                               threshold=threshold))
    elif args.kind == 'correlation':
        XtX = gram(data.train)
        for target, (threshold, alpha) in CORRELATION_THRESHOLDS[args.dataset].items():
            A = correlation_pattern(XtX, data.train.shape[0], alpha, threshold, max_in_col=data.n_items)
            model = Fixed(restrict_to_pattern(edlae.B, A))
            rows.append(record(model, sparse_predictor(model.B), target_density=target, density=model.density,
                               threshold=threshold, alpha=alpha))

out = Path('results')
out.mkdir(exist_ok=True)
suffix = f'_{args.method}' if args.kind == 'lowrank' else ''
path = out / f'{args.kind}{suffix}_{args.dataset}.csv'
pd.DataFrame(rows).to_csv(path, index=False)
print(f'saved {path}')
