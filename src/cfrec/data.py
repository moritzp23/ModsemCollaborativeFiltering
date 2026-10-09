"""Download, preprocess and load the benchmark datasets.

Preprocessing follows the protocol of Liang et al. (Mult-VAE) / Rendle et al. (iALS benchmarks),
adapted from https://github.com/google-research/google-research/blob/master/ials/vae_benchmarks/generate_data.py
(Copyright 2022 The Google Research Authors, Apache License 2.0).

Strong generalization: users are split into disjoint train / validation / test sets. For validation
and test users, 80% of their interactions are given to the model (``*_tr``, fold-in) and the remaining
20% are held out for evaluation (``*_te``).
"""
from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

SEED = 98765

DATASETS = {
    'ml-20m': dict(n_heldout_users=10_000, min_uc=5, min_sc=0),
    'netflix': dict(n_heldout_users=40_000, min_uc=5, min_sc=0),
    'msd': dict(n_heldout_users=50_000, min_uc=20, min_sc=200),
}

ML20M_URL = 'https://files.grouplens.org/datasets/movielens/ml-20m.zip'
MSD_URL = 'http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip'
# Netflix-Prize data can no longer be downloaded from netflixprize.com, get it from Kaggle:
# https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

SPLITS = ('train', 'validation_tr', 'validation_te', 'test_tr', 'test_te')


# --------------------------------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------------------------------

def _count(tp: pd.DataFrame, col: str) -> pd.Series:
    return tp[[col]].groupby(col, as_index=True).size()


def filter_triplets(tp: pd.DataFrame, min_uc: int, min_sc: int):
    """Keep items with >= min_sc users, then users with >= min_uc items."""
    if min_sc > 0:
        itemcount = _count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]
    if min_uc > 0:
        usercount = _count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]
    return tp, _count(tp, 'userId'), _count(tp, 'movieId')


def split_train_test_proportion(data: pd.DataFrame, test_prop: float = 0.2):
    """Per user (with >= 5 interactions), hold out a random `test_prop` fraction of interactions."""
    tr_list, te_list = [], []
    np.random.seed(SEED)
    for _, group in data.groupby('userId'):
        n_items_u = len(group)
        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)
    return pd.concat(tr_list), pd.concat(te_list)


def generate_data(raw_data: pd.DataFrame, output_dir: str | Path, n_heldout_users: int, min_uc: int, min_sc: int):
    """Filter, split by user and write train / validation / test csv files with columns (uid, sid)."""
    raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc, min_sc)
    sparsity = raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    print(f'After filtering, there are {raw_data.shape[0]} interactions from {user_activity.shape[0]} users '
          f'and {item_popularity.shape[0]} items (sparsity: {sparsity * 100:.3f}%)')

    unique_uid = user_activity.index
    np.random.seed(SEED)
    unique_uid = unique_uid[np.random.permutation(unique_uid.size)]
    n_users = unique_uid.size
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2):(n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
    unique_sid = pd.unique(train_plays['movieId'])
    show2id = {sid: i for i, sid in enumerate(unique_sid)}
    profile2id = {pid: i for i, pid in enumerate(unique_uid)}

    def numerize(tp):
        return pd.DataFrame({'uid': tp['userId'].map(profile2id).values,
                             'sid': tp['movieId'].map(show2id).values})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'unique_sid.txt', 'w') as f:
        f.writelines(f'{sid}\n' for sid in unique_sid)

    def heldout(users):
        plays = raw_data.loc[raw_data['userId'].isin(users)]
        return split_train_test_proportion(plays.loc[plays['movieId'].isin(unique_sid)])

    vad_tr, vad_te = heldout(vd_users)
    test_tr, test_te = heldout(te_users)
    for name, plays in zip(SPLITS, (train_plays, vad_tr, vad_te, test_tr, test_te)):
        numerize(plays).to_csv(output_dir / f'{name}.csv', index=False)


def _download(url: str, dest: Path):
    if not dest.exists():
        print(f'Downloading {url}')
        urllib.request.urlretrieve(url, dest)


def prepare_ml20m(root: str | Path = 'data'):
    root = Path(root)
    zip_path = root / 'raw' / 'ml-20m.zip'
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    _download(ML20M_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf, zf.open('ml-20m/ratings.csv') as f:
        raw_data = pd.read_csv(f, header=0)
    raw_data = raw_data[raw_data['rating'] > 3.5]  # binarize: keep ratings >= 4
    generate_data(raw_data, root / 'ml-20m', **DATASETS['ml-20m'])


def prepare_msd(root: str | Path = 'data'):
    root = Path(root)
    zip_path = root / 'raw' / 'msd.zip'
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    _download(MSD_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf, zf.open('train_triplets.txt') as f:
        raw_data = pd.read_csv(f, sep='\t', header=None, names=['userId', 'movieId', 'count'])
    generate_data(raw_data, root / 'msd', **DATASETS['msd'])


def prepare_netflix(netflix_zip: str | Path, root: str | Path = 'data'):
    """`netflix_zip` is the archive from Kaggle containing combined_data_{1..4}.txt."""
    root = Path(root)
    parts = []
    with zipfile.ZipFile(netflix_zip) as zf:
        for i in range(1, 5):
            movie_ids, user_ids, ratings = [], [], []
            with zf.open(f'combined_data_{i}.txt') as f:
                for line in map(bytes.decode, f):
                    line = line.strip()
                    if line.endswith(':'):
                        movie_id = int(line[:-1])
                    else:
                        user_id, rating, _ = line.split(',')
                        movie_ids.append(movie_id)
                        user_ids.append(int(user_id))
                        ratings.append(float(rating))
            parts.append(pd.DataFrame({'movieId': movie_ids, 'userId': user_ids, 'rating': ratings}))
    raw_data = pd.concat(parts, ignore_index=True)
    raw_data = raw_data.sort_values(by=['userId', 'movieId']).reset_index(drop=True)
    raw_data = raw_data[raw_data['rating'] > 3.5]
    generate_data(raw_data, root / 'netflix', **DATASETS['netflix'])


# --------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------

def to_csr(df: pd.DataFrame, n_items: int, users=None) -> sp.csr_matrix:
    """Binary user x item matrix. Rows follow `users` (default: sorted unique uids of `df`)."""
    if users is None:
        users = np.unique(df['uid'].values)
    rows = np.searchsorted(users, df['uid'].values)
    X = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (rows, df['sid'].values)),
                      shape=(len(users), n_items), dtype=np.float32)
    X.sum_duplicates()
    return X


@dataclass
class Dataset:
    name: str
    train: sp.csr_matrix
    val_tr: sp.csr_matrix
    val_te: sp.csr_matrix
    test_tr: sp.csr_matrix
    test_te: sp.csr_matrix

    @property
    def n_items(self) -> int:
        return self.train.shape[1]

    def __repr__(self):
        return (f'Dataset({self.name}: {self.n_items} items, {self.train.shape[0]} train users '
                f'({self.train.nnz} interactions), {self.val_tr.shape[0]} val users, {self.test_tr.shape[0]} test users)')


def load_dataset(name: str, root: str | Path = 'data') -> Dataset:
    """Load preprocessed csv files. Fold-in (`*_tr`) and held-out (`*_te`) matrices share row order."""
    path = Path(root) / name
    if not (path / 'train.csv').exists():
        raise FileNotFoundError(f'{path}/train.csv not found, run `python scripts/prepare_data.py {name}` first.')
    n_items = sum(1 for _ in open(path / 'unique_sid.txt'))
    dfs = {s: pd.read_csv(path / f'{s}.csv') for s in SPLITS}

    def paired(tr, te):
        users = np.union1d(tr['uid'].values, te['uid'].values)
        return to_csr(tr, n_items, users), to_csr(te, n_items, users)

    val_tr, val_te = paired(dfs['validation_tr'], dfs['validation_te'])
    test_tr, test_te = paired(dfs['test_tr'], dfs['test_te'])
    return Dataset(name, to_csr(dfs['train'], n_items), val_tr, val_te, test_tr, test_te)
