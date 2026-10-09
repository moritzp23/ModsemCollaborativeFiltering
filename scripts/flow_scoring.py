"""Compare ranking by the one-step posterior mean E[x | y] with averages of S posterior samples (ODE).

    python scripts/flow_scoring.py ml-20m --load results/fm_ml-20m.pkl --samples 1 4 16 --steps 5 10 20

Without --load, trains the flow-matching model with the configuration in cfrec.configs and saves it.
"""
import argparse
import json
import time
from pathlib import Path

from cfrec.configs import BEST_PARAMS
from cfrec.data import load_dataset
from cfrec.evaluation import evaluate, format_result
from cfrec.models import FMRecommender

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('dataset', choices=['ml-20m', 'netflix', 'msd'])
parser.add_argument('--split', choices=['test', 'val'], default='val')
parser.add_argument('--samples', type=int, nargs='+', default=[1, 4, 16])
parser.add_argument('--steps', type=int, nargs='+', default=[5, 10, 20])
parser.add_argument('--load', help='FMRecommender checkpoint (fmbayes-style pickle)')
parser.add_argument('--root', default='data')
args = parser.parse_args()

data = load_dataset(args.dataset, args.root)
if args.load:
    model = FMRecommender.load(args.load, verbose=False)
else:
    model = FMRecommender(**{**BEST_PARAMS['fm'][args.dataset], 'verbose': False})
    model.fit(data.train, data.val_tr, data.val_te)
    print(f'fit: {model.fit_time:.0f}s, saved {model.save(f"results/fm_{args.dataset}.pkl")}')

X_in, X_out = (data.test_tr, data.test_te) if args.split == 'test' else (data.val_tr, data.val_te)
rows = []


def run(n_samples, n_steps=None):
    model.score_samples, model.steps = n_samples, n_steps or model.steps
    start = time.perf_counter()
    res = evaluate(model, X_in, X_out, batch_size=model.score_batch)
    elapsed = time.perf_counter() - start
    info = {'scoring': 'one-step mean'} if n_samples == 0 else {'samples': n_samples, 'steps': n_steps}
    print(info, format_result(res), f'({elapsed:.1f}s)', flush=True)
    rows.append({**info, **res, 'eval_time': elapsed})


run(0)
for n_steps in args.steps:
    for n_samples in args.samples:
        run(n_samples, n_steps)

out = Path('results') / f'flow_scoring_{args.dataset}_{args.split}.json'
out.write_text(json.dumps(rows, indent=2))
print(f'saved {out}')
