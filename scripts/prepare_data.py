"""Download and preprocess benchmark datasets.

    python scripts/prepare_data.py ml-20m
    python scripts/prepare_data.py msd
    python scripts/prepare_data.py netflix --netflix-zip ~/Downloads/archive.zip
"""
import argparse

from cfrec.data import load_dataset, prepare_ml20m, prepare_msd, prepare_netflix

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('dataset', choices=['ml-20m', 'msd', 'netflix'])
parser.add_argument('--root', default='data')
parser.add_argument('--netflix-zip', help='Kaggle netflix-prize-data archive')
args = parser.parse_args()

if args.dataset == 'ml-20m':
    prepare_ml20m(args.root)
elif args.dataset == 'msd':
    prepare_msd(args.root)
else:
    if not args.netflix_zip:
        parser.error('--netflix-zip is required for netflix')
    prepare_netflix(args.netflix_zip, args.root)

print(load_dataset(args.dataset, args.root))
