"""Compare ranking by the one-step posterior mean E[x | y] with averages of K posterior samples (ODE).

    python scripts/flow_scoring.py ml-20m --samples 1 4 16 --steps 5 10 20
"""
import argparse
import json
import time
from pathlib import Path

import torch

from cfrec.configs import BEST_PARAMS
from cfrec.data import load_dataset
from cfrec.evaluation import evaluate, format_result
from cfrec.models import FlowMatchingCF

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('dataset', choices=['ml-20m', 'netflix', 'msd'])
parser.add_argument('--split', choices=['test', 'val'], default='val')
parser.add_argument('--samples', type=int, nargs='+', default=[1, 4, 16])
parser.add_argument('--steps', type=int, nargs='+', default=[5, 10, 20])
parser.add_argument('--load', help='checkpoint (state dict) to evaluate instead of training')
parser.add_argument('--root', default='data')
args = parser.parse_args()

data = load_dataset(args.dataset, args.root)
model = FlowMatchingCF(**{**BEST_PARAMS['flow'][args.dataset], 'verbose': False})
if args.load:
    model.build(data.n_items)
    model.net.load_state_dict(torch.load(args.load, map_location=model.device))
else:
    model.fit(data.train, data.val_tr, data.val_te)
    ckpt = Path('results') / f'flow_{args.dataset}.pt'
    torch.save(model.net.state_dict(), ckpt)
    print(f'fit: {model.fit_time:.0f}s, saved {ckpt}')

X_in, X_out = (data.test_tr, data.test_te) if args.split == 'test' else (data.val_tr, data.val_te)
rows = []


def run(**kw):
    for k, v in kw.items():
        setattr(model, k, v)
    torch.manual_seed(0)
    start = time.perf_counter()
    res = evaluate(model, X_in, X_out, batch_size=500)
    elapsed = time.perf_counter() - start
    print(kw, format_result(res), f'({elapsed:.1f}s)', flush=True)
    rows.append({**kw, **res, 'eval_time': elapsed})


run(score_mode='mean')
for n_steps in args.steps:
    for n_samples in args.samples:
        run(score_mode='ode', n_samples=n_samples, n_steps=n_steps)

out = Path('results') / f'flow_scoring_{args.dataset}_{args.split}.json'
out.write_text(json.dumps(rows, indent=2))
print(f'saved {out}')
