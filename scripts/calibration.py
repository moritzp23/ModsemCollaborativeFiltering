"""Calibration of P(x_i = 1 | y) and of the number of hidden items, on the test users.

    python scripts/calibration.py ml-20m --flow k05=results/flow_ml-20m.pt k08=results/flow_ml-20m_k08.pt

Models: EASE + Platt scaling (fitted on validation users), and each flow checkpoint, raw and + Platt.
For the flows, count intervals from joint posterior samples are compared with intervals from independent
Bernoulli marginals. Writes results/calibration_<dataset>.json and a reliability-diagram png.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from cfrec.calibration import CalibrationAccumulator, fit_platt, sample_count_coverage
from cfrec.configs import BEST_PARAMS
from cfrec.data import load_dataset
from cfrec.models import EASE, FlowMatchingCF

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('dataset', choices=['ml-20m', 'netflix', 'msd'])
parser.add_argument('--flow', nargs='+', default=[], metavar='NAME=CKPT',
                    help='flow checkpoints; keep_prob is read from NAME if it is of the form k08 / k05')
parser.add_argument('--n-samples', type=int, default=64)
parser.add_argument('--n-steps', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--root', default='data')
args = parser.parse_args()

data = load_dataset(args.dataset, args.root)
X_in, X_out = data.test_tr, data.test_te
keep = np.diff(X_out.indptr) > 0
X_in, X_out = X_in[keep], X_out[keep]


def evaluate_probs(prob_fn):
    acc = CalibrationAccumulator()
    for start in range(0, X_in.shape[0], args.batch_size):
        sl = slice(start, start + args.batch_size)
        acc.update(prob_fn(X_in[sl]), X_in[sl], X_out[sl])
    return acc


results, curves = {}, {}


def report(name, acc, extra=None):
    res = {**acc.summary(), **(extra or {})}
    results[name] = res
    curves[name] = {k: v.tolist() for k, v in acc.reliability('all').items()}
    print(f"{name:14s} ECE {res['ece_all']:.2e} | ECE@top100 {res['ece_top100']:.3f} | "
          f"Brier@top100 {res['brier_top100']:.4f} | NLL {res['nll_all']:.4f} | "
          f"count bias {res['count_bias']:+.1f}, cover50/90 (indep) "
          f"{res['count_cover50_indep']:.2f}/{res['count_cover90_indep']:.2f}"
          + (f" (samples) {res['count_cover50_samples']:.2f}/{res['count_cover90_samples']:.2f}"
             if 'count_cover50_samples' in res else ''), flush=True)


# ---- EASE + Platt scaling
ease = EASE(**BEST_PARAMS['ease'][args.dataset]).fit(data.train)
a, b = fit_platt(ease.score, data.val_tr, data.val_te)
print(f'EASE Platt: a={a:.3f}, b={b:.3f}')
report('EASE+Platt', evaluate_probs(lambda X: expit(a * ease.score(X) + b)))

# ---- flows
for spec in args.flow:
    name, ckpt = spec.split('=')
    params = dict(BEST_PARAMS['flow'][args.dataset], verbose=False, batch_size=args.batch_size)
    if name.startswith('k') and name[1:].isdigit():
        params['keep_prob'] = int(name[1:]) / 10
    model = FlowMatchingCF(**params).build(data.n_items)
    model.net.load_state_dict(torch.load(ckpt, map_location=model.device))
    torch.manual_seed(0)

    def probs(X):
        return model.posterior_mean(torch.as_tensor(X.toarray(), device=model.device)).cpu().numpy()

    def samples(X, n):
        return model.sample(torch.as_tensor(X.toarray(), device=model.device), n_samples=n,
                            n_steps=args.n_steps).cpu().numpy()

    extra = sample_count_coverage(samples, X_in, X_out, args.n_samples, batch_size=100)
    report(f'flow-{name}', evaluate_probs(probs), extra)

    # Platt on the logit of the posterior mean
    def logit_probs(X):
        p = np.clip(probs(X), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    a, b = fit_platt(logit_probs, data.val_tr, data.val_te)
    print(f'flow-{name} Platt: a={a:.3f}, b={b:.3f}')
    report(f'flow-{name}+Platt', evaluate_probs(lambda X: expit(a * logit_probs(X) + b)))

out = Path('results')
(out / f'calibration_{args.dataset}.json').write_text(json.dumps({'metrics': results, 'reliability': curves}, indent=2))

# ---- reliability diagram (palette: categorical slots 1-4, fixed order)
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
fig, ax = plt.subplots(figsize=(6.5, 5.5))
lims = (1e-5, 1)
ax.plot(lims, lims, color='#9a9a94', lw=1, ls='--', zorder=1)
for color, (name, c) in zip(COLORS, curves.items()):
    ax.plot(c['mean_pred'], c['freq'], color=color, lw=2, marker='o', ms=6, label=name, zorder=2,
            markeredgecolor='white', markeredgewidth=1)
ax.set(xscale='log', yscale='log', xlim=lims, ylim=lims, xlabel='mean predicted probability',
       ylabel='observed frequency of held-out items',
       title=f'Reliability on {args.dataset} test users (unobserved items)')
ax.grid(True, which='major', color='#e6e6e1', lw=0.8)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc='upper left')
fig.tight_layout()
fig.savefig(out / f'reliability_{args.dataset}.png', dpi=150)
print(f"saved {out / f'calibration_{args.dataset}.json'} and reliability_{args.dataset}.png")
