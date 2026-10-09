"""Fit a model on the training users and evaluate on the test (or validation) users.

    python scripts/run.py ease ml-20m                    # best hyperparameters from cfrec.configs
    python scripts/run.py ease ml-20m --set lmbda=300    # override hyperparameters
    python scripts/run.py edlae ml-20m --split val
"""
import argparse
import ast
import inspect
import json
import time
from pathlib import Path

from cfrec.configs import BEST_PARAMS
from cfrec.data import load_dataset
from cfrec.evaluation import evaluate, format_result
from cfrec.models import MODELS


def parse_overrides(pairs):
    out = {}
    for pair in pairs:
        key, value = pair.split('=', 1)
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('model', choices=sorted(MODELS))
parser.add_argument('dataset', choices=['ml-20m', 'netflix', 'msd'])
parser.add_argument('--split', choices=['test', 'val'], default='test')
parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='hyperparameter overrides')
parser.add_argument('--root', default='data')
parser.add_argument('--out', default='results', help='directory for the json result')
parser.add_argument('--tag', default='', help='suffix for the result file name')
parser.add_argument('--save', help='save the trained model (fm: fmbayes-style pickle; torch: state dict)')
args = parser.parse_args()

params = {**BEST_PARAMS.get(args.model, {}).get(args.dataset, {}), **parse_overrides(args.set)}
data = load_dataset(args.dataset, args.root)
print(data)
print(f'{args.model}({params})')

model = MODELS[args.model](**params)
start = time.perf_counter()
if 'X_val_in' in inspect.signature(model.fit).parameters:  # model selection on the validation users
    model.fit(data.train, data.val_tr, data.val_te)
else:
    model.fit(data.train)
fit_time = time.perf_counter() - start
print(f'fit: {fit_time:.1f}s')

X_in, X_out = (data.test_tr, data.test_te) if args.split == 'test' else (data.val_tr, data.val_te)
start = time.perf_counter()
result = evaluate(model, X_in, X_out)
eval_time = time.perf_counter() - start
print(f'evaluate ({args.split}): {eval_time:.1f}s')
print(format_result(result))

out = Path(args.out)
out.mkdir(exist_ok=True)
record = dict(model=args.model, dataset=args.dataset, split=args.split, params=params,
              fit_time=fit_time, eval_time=eval_time, **result)
if hasattr(model, 'history'):
    record['history'] = model.history
tag = f'_{args.tag}' if args.tag else ''
path = out / f'{args.model}_{args.dataset}_{args.split}{tag}.json'
path.write_text(json.dumps(record, indent=2))
if args.save:
    if hasattr(model, 'save'):
        model.save(args.save)
    else:
        import torch
        torch.save(model.net.state_dict(), args.save)
    print(f'saved model {args.save}')
print(f'saved {path}')
