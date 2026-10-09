# Fast Methods for Collaborative Filtering

Code for the modeling seminar report *Fast Methods for Collaborative Filtering* (M. Poguntke, TU Chemnitz, 2023):
linear autoencoders (EASE, DLAE, EDLAE), their sparse and low-rank approximations, Markov random fields,
ADMM-SLIM, ItemKNN and RecVAE, evaluated with the strong-generalization protocol of Liang et al. (2018)
on MovieLens-20M, Netflix and the Million Song Dataset.

## Setup

```bash
conda env create -f environment.yml     # creates env `cfrec`, installs this package in editable mode
conda activate cfrec
pip install -e ".[vae]"                 # optional: PyTorch for RecVAE
```

(`requirements.txt` lists the same dependencies for plain pip.)

## Data

```bash
python scripts/prepare_data.py ml-20m                                   # downloads from grouplens
python scripts/prepare_data.py msd                                      # ~500 MB download
python scripts/prepare_data.py netflix --netflix-zip ~/Downloads/archive.zip   # Kaggle netflix-prize-data
```

Preprocessed csv files go to `data/<dataset>/` (train, validation_tr/te, test_tr/te). The splits are
identical to the ones used in the report (same filtering, seeds and ordering).

| dataset | items  | users   | held-out users (val / test) |
|---------|--------|---------|-----------------------------|
| ml-20m  | 20,108 | 136,677 | 10,000 / 10,000             |
| netflix | 17,769 | 463,435 | 40,000 / 40,000             |
| msd     | 41,140 | 571,355 | 50,000 / 50,000             |

## Running models

```bash
python scripts/run.py ease ml-20m                      # best hyperparameters from cfrec/configs.py
python scripts/run.py edlae netflix --set lmbda=400    # override hyperparameters
python scripts/run.py recvae ml-20m                    # GPU recommended
```

Results (Recall@20, Recall@50, NDCG@100 with standard errors) are printed and saved to `results/`.

Approximations of EDLAE (report Section 3.3 / 3.4):

```bash
python scripts/approximations.py ml-20m lowrank --method eig --ranks 100 1000 5000 --timing
python scripts/approximations.py ml-20m magnitude      # keep |B_ij| > t
python scripts/approximations.py ml-20m correlation    # pattern from thresholded correlation matrix
python scripts/approximations.py ml-20m mrf            # sparse MRF approximation sweep
```

Conditional flow matching (CF as a Bayesian inverse problem, `models/flow.py`): learns the posterior
p(x | y) of a user's full interaction vector x given an observed subset y, amortized over users. Ranking
uses the one-step posterior mean E[x | y]; posterior samples come from integrating the flow ODE.

```bash
python scripts/run.py flow ml-20m                       # train (model selection on validation users)
python scripts/flow_scoring.py ml-20m                   # posterior mean vs. averaged ODE samples
```

The same approach built on `fmbayes` (`conditional-flows-UQ/fmbayes`) (JAX; conditional flow matching for Bayesian
inverse problems) lives in `models/fm.py`: fmbayes' network registry, training loop (`flows.train.fit`, the
masking forward operator as its `y_fn`) and ODE transport, plus CF extensions in the same style
(`denoiser-gated`, `affine-tgated`, `gauss-ease`: fmbayes' Gaussian head with a low-rank EASE mean).
It needs its own environment (Python >= 3.13, jax 0.6):

```bash
conda env create -f environment-fm.yml      # env `cfrec-fm`
python scripts/run.py fm ml-20m --set velocity_param=denoiser-gated
```

From Python:

```python
from cfrec.data import load_dataset
from cfrec.models import EDLAE
from cfrec.evaluation import evaluate

data = load_dataset('ml-20m')
model = EDLAE(lmbda=300, p=0.33).fit(data.train)
evaluate(model, data.test_tr, data.test_te)
```

## Layout

```
src/cfrec/
  data.py          download / preprocessing / loading (csr matrices, aligned fold-in & held-out rows)
  metrics.py       vectorized Recall@k, NDCG@k, top-k selection
  evaluation.py    strong-generalization evaluation with standard errors
  timing.py        inference latency (batch of 1000 users / single query)
  linalg.py        fast SPD inverse (Cholesky + LAPACK potri), Gram matrix
  sparse.py        sparsity patterns (magnitude, thresholded correlation)
  configs.py       tuned hyperparameters from the report
  models/
    popularity.py  MostPopular
    itemknn.py     ItemKNN (cosine / Pearson)
    linear.py      EASE, EDLAE, DLAE, MRFDense
    mrf.py         MRFApprox (sparse MRF, block-wise inverses)
    admm.py        ADMMSlim (sparse EDLAE / SLIM via ADMM)
    lowrank.py     SVD / eigen low-rank approximations
    recvae.py      RecVAE
    flow.py        conditional flow matching (amortized posterior p(x | y))
scripts/           prepare_data.py, run.py, approximations.py
tests/             unit tests (pytest)
legacy/            original notebooks and Models.py from the report, kept for reference
```

All models implement `fit(X_train)` and `score(X_fold_in) -> dense scores`; masking of seen items and
top-k selection live in `cfrec.evaluation`.

iALS results in the report were produced with Google's C++ implementation
([google-research/ials](https://github.com/google-research/google-research/tree/master/ials)) and are not part of this package.

## Changes compared to the report code (`legacy/`)

- Fold-in and held-out matrices are built on a shared user index; the notebook's standard-error cells
  parsed them separately, which misaligns rows if a test user has no held-out items.
- ADMM: the soft-thresholding step thresholded `B + Gamma/rho + lambda1/rho^2` instead of `B + Gamma/rho`
  (see `models/admm.py`); fixed.
- `DLAE(method='inv')` referenced an undefined function; fixed. The LU variant was dropped (equivalent to Cholesky).
- `MRFApprox.fit_old` and the duplicated helpers of every model class were removed.
