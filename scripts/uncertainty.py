"""Does the flow's joint posterior add uncertainty information beyond its marginal probabilities?

    python scripts/uncertainty.py ml-20m --model flow=results/ckpt/a.pkl t0=results/ckpt/b.pkl \
        --no-samples t0 --steps 10 20

For every flow-matching checkpoint (test users, unobserved items):
  - ranking metrics and marginal calibration of the one-step posterior mean p = E[x | y];
  - joint scores (CRPS / coverage of the number of hidden items and of top-10 hits, energy score) for
      * 'indep':   independent Bernoulli(p) draws -- the marginals alone,
      * 'flow-K':  flow posterior samples (Euler, K steps, thresholded at 0.5),
    so flow-K vs indep isolates the dependence structure the flow learned;
  - sample diagnostics (near-binary? does the sample mean match p?).
EASE + Platt (fitted on validation users) with independent draws is the classical baseline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from cfrec.calibration import CalibrationAccumulator, JointAccumulator, fit_platt, sample_diagnostics
from cfrec.configs import BEST_PARAMS
from cfrec.data import load_dataset
from cfrec.evaluation import evaluate
from cfrec.models import EASE, FMRecommender

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('dataset', choices=['ml-20m', 'netflix', 'msd'])
parser.add_argument('--model', nargs='+', default=[], metavar='NAME=CKPT')
parser.add_argument('--no-samples', nargs='*', default=[], metavar='NAME',
                    help='models scored with independent draws only (e.g. the t = 0 ablation, whose flow is untrained)')
parser.add_argument('--n-samples', type=int, default=64)
parser.add_argument('--steps', type=int, nargs='+', default=[10, 20])
parser.add_argument('--grid', nargs='+', default=['uniform'], choices=['uniform', 'graded'])
parser.add_argument('--users', type=int, default=None, help='use only the first N test users')
parser.add_argument('--batch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--root', default='data')
args = parser.parse_args()

data = load_dataset(args.dataset, args.root)
keep = np.diff(data.test_te.indptr) > 0
X_in, X_out = data.test_tr[keep], data.test_te[keep]
if args.users:
    X_in, X_out = X_in[:args.users], X_out[:args.users]
print(f'{X_in.shape[0]} test users, {args.n_samples} samples per user', flush=True)
results = {}


def run(name, prob_fn, sample_fns):
    rng = np.random.default_rng(args.seed)
    cal = CalibrationAccumulator()
    joint = {'indep': JointAccumulator(), **{k: JointAccumulator() for k in sample_fns}}
    diag = {k: [] for k in sample_fns}
    for start in range(0, X_in.shape[0], args.batch):
        xi, xo = X_in[start:start + args.batch], X_out[start:start + args.batch]
        p = np.clip(prob_fn(xi), 1e-6, 1 - 1e-6)
        unobs = xi.toarray() == 0
        cal.update(p, xi, xo)
        joint['indep'].update((rng.random((args.n_samples, *p.shape)) < p) & unobs, p, xi, xo)
        for k, fn in sample_fns.items():
            S = fn(xi)
            diag[k].append(sample_diagnostics(S, p, unobs))
            joint[k].update((S > 0.5) & unobs, p, xi, xo)
    res = {'calibration': cal.summary(), 'joint': {k: v.summary() for k, v in joint.items()},
           'diagnostics': {k: {m: float(np.mean([d[m] for d in v])) for m in v[0]} for k, v in diag.items()}}
    results[name] = res
    c = res['calibration']
    print(f"\n== {name}: ECE@top100 {c['ece_top100']:.3f}, Brier@top100 {c['brier_top100']:.4f}, "
          f"count bias {c['count_bias']:+.1f}", flush=True)
    print(f"   {'':10s} {'CRPS N':>8s} {'cov50/90 N':>11s} {'CRPS H10':>9s} {'cov50/90 H10':>13s} {'energy':>8s}")
    for k, j in res['joint'].items():
        print(f"   {k:10s} {j['crps_count']:8.2f} {j['cover50_count']:5.2f}/{j['cover90_count']:.2f} "
              f"{j['crps_hits']:9.3f} {j['cover50_hits']:7.2f}/{j['cover90_hits']:.2f} {j['energy']:8.3f}")
    for k, d in res['diagnostics'].items():
        print(f"   {k}: non-binary {d['frac_nonbinary']:.4f}, |mean(samples) - p| {d['mean_abs_gap_to_onestep']:.5f}")


# classical baseline: EASE + Platt scaling, independent draws
ease = EASE(**BEST_PARAMS['ease'][args.dataset]).fit(data.train)
a, b = fit_platt(ease.score, data.val_tr, data.val_te)
run('EASE+Platt', lambda X: expit(a * ease.score(X) + b), {})

for spec in args.model:
    name, ckpt = spec.split('=')
    model = FMRecommender.load(ckpt, verbose=False)
    res_rank = evaluate(model, X_in, X_out)
    def sampler(k, g):
        def f(X):
            model.grid = g
            return model.sample(X, args.n_samples, steps=k)
        return f
    samplers = {} if name in args.no_samples else {
        f'flow-{k}' + ('g' if g == 'graded' else ''): sampler(k, g) for k in args.steps for g in args.grid}
    run(name, model.posterior_mean, samplers)
    results[name]['ranking'] = {m: v for m, v in res_rank.items() if '@' in m}
    print(f"   ranking: " + ', '.join(f'{m} {v:.4f}' for m, v in res_rank.items() if '@' in m and '_se' not in m))

out = Path('results') / f'uncertainty_{args.dataset}.json'
out.write_text(json.dumps({'args': vars(args), 'results': results}, indent=2))
print(f'\nsaved {out}')
