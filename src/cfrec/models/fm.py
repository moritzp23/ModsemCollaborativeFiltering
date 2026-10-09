"""Bayesian conditional flow matching for collaborative filtering, built on `fmbayes`.

Inverse-problem view: x in {0,1}^I is a user's full interaction vector, the observation y = M(x) keeps each
positive with probability `keep_prob`. Joint samples (x, y) come from the training users; the forward
operator is applied per batch through `fmbayes.flows.train.fit`'s `y_fn` hook (the conditioning array is x
itself, `yg = xg`).

Uses from `fmbayes`: the velocity-network registry (`flows.models`), the training loop (`flows.train.fit`),
the flow-matching loss and the ODE transport (`flows.integrators.make_flow_transport`, density off: the exact
Jacobian trace is infeasible at d ~ 20k). CF-specific extensions follow the same conventions (input
`[x_t, y, t]`, zero-initialised learned part) and are registered in `MODEL_MAP` below.

Posterior mean: for the independent coupling, x0 + v(0, x0; y) = E[x1 | y] for any x0
(notes/one_step_mean).
"""
from __future__ import annotations

import os

# the dense training matrix (~9 GB for ML-20M) lives on the device: allow more than JAX's default 75%
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.95')

import inspect
import pickle
import time
from dataclasses import dataclass, field
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from fmbayes.flows import models as fm_models
from fmbayes.flows import train as fm_train
from fmbayes.flows.integrators import make_flow_transport
from fmbayes.flows.networks import NNAffine, NNGaussian, draw_reference_jax

from .base import Recommender


# ------------------------------------------------------------------------------------------------
# CF extensions of the fmbayes network family
# ------------------------------------------------------------------------------------------------

class NNDenoiserGated(NNAffine):
    """v = (D - x_t) / (1 - t) with the data prediction D = sigmoid(MLP([t x_t, y, t])).

    `D` estimates E[x1 | x_t, y] for binary x1. The state enters the MLP as t * x_t: at t = 0 the
    optimal D is independent of x_0 (= E[x1 | y]), and the gate makes that exact by construction.
    `t_max` caps 1/(1-t) so the velocity stays finite at t -> 1. Train with `denoiser_bce_loss`.
    """
    t_max: float = 0.99
    zero_init_core = False

    def denoise_logits(self, inputs):
        xt, y, t = self._split(inputs)
        return self.core(jnp.concatenate([t * xt, y, t], axis=-1))

    def __call__(self, inputs):
        xt, _, t = self._split(inputs)
        d = nn.sigmoid(self.denoise_logits(inputs))
        return (d - xt) / (1.0 - jnp.minimum(t, self.t_max))


class NNAffineTGated(NNAffine):
    """v = a(t) x_t + MLP([t x_t, y, t]): fmbayes' affine skip and loss, with the state time-gated.

    At t = 0, a(0) = -1, so x_0 + v(0, x_0; y) = MLP([0, y, 0]): the one-step posterior mean is
    independent of the reference draw by construction (cf. `NNDenoiserGated`).
    """

    def _corr(self, inputs, xt, y, t):
        return self.core(jnp.concatenate([t * xt, y, t], axis=-1))


class NNGaussEASE(NNGaussian):
    """fmbayes' Gaussian head with a low-rank EASE mean:

        v = alpha(t) x_t + (1 - t alpha(t)) m(y) + MLP([t x_t, y / |y|, t]),
        m(y) = mean_scale * (y U) V^T + obs_scale * y + mean_bias.

    For a linear-Gaussian inverse problem the posterior mean is linear in the data (`NNGaussian`'s linear
    head); in CF the linear autoencoder EASE/EDLAE is such a linear posterior mean (Steck's Gaussian MRF).
    U V^T is its rank-`rank` approximation (B ~ B V_k V_k^T), the full 20k x 20k head would not fit.
    `obs_scale * y` maps observed items to x = 1 (B has a zero diagonal). y is the raw binary observation;
    the MLP sees it l2-normalised and the state time-gated, so the one-step posterior mean is
    m(y) + MLP([0, y/|y|, 0]) for every x_0. All mean parameters, s^2 and the MLP are trained with the
    standard flow-matching loss; U, V, s^2 and the scales are initialised through `fit(params0=...)`
    (`ease_params0`), so the model starts at the (low-rank) EDLAE posterior mean.
    """
    rank: int = 1000
    freeze_lowrank: bool = False    # keep U, V at the EDLAE factors (use with weight_decay = 0)

    def setup(self):
        NNAffine.setup(self)        # the core MLP; no dense (x_dim x x_dim) mean head
        self.log_s2 = self.param('log_s2', nn.initializers.zeros, (self.x_dim,), self.dtype)
        self.U = self.param('U', nn.initializers.zeros, (self.x_dim, self.rank), self.dtype)
        self.V = self.param('V', nn.initializers.zeros, (self.x_dim, self.rank), self.dtype)
        self.mean_scale = self.param('mean_scale', nn.initializers.ones, (), self.dtype)
        self.obs_scale = self.param('obs_scale', nn.initializers.ones, (), self.dtype)
        self.mean_bias = self.param('mean_bias', nn.initializers.zeros, (self.x_dim,), self.dtype)

    def _mean(self, y):
        U, V = self.U, self.V
        if self.freeze_lowrank:
            U, V = jax.lax.stop_gradient(U), jax.lax.stop_gradient(V)
        return self.mean_scale * ((y @ U) @ V.T) + self.obs_scale * y + self.mean_bias

    def _corr(self, inputs, xt, y, t, m):
        return self.core(jnp.concatenate([t * xt, l2_rows(y), t], axis=-1))


class NNDenoiserEASE(NNDenoiserGated):
    """`NNDenoiserGated` with a frozen low-rank EASE skip in the logits of the data prediction:

        D = sigmoid(ease_scale * (y U) V^T + obs_logit * y + item_bias + MLP([t x_t, y / |y|, t])).

    The same idea as `NNGaussEASE` (the linear posterior mean as a built-in skip), but trained with the BCE
    data-prediction loss. y is the raw binary observation. U, V are the EDLAE factors and get no gradient;
    the MLP is zero-initialised, so training starts at a calibrated EASE prediction (`ease_params0`).
    Weight decay shrinks U V^T uniformly, which `ease_scale` absorbs.
    """
    rank: int = 1000
    freeze_lowrank: bool = True
    zero_init_core = True

    def setup(self):
        NNAffine.setup(self)
        self.U = self.param('U', nn.initializers.zeros, (self.x_dim, self.rank), self.dtype)
        self.V = self.param('V', nn.initializers.zeros, (self.x_dim, self.rank), self.dtype)
        self.ease_scale = self.param('ease_scale', nn.initializers.constant(5.0), (), self.dtype)
        self.obs_logit = self.param('obs_logit', nn.initializers.constant(10.0), (), self.dtype)
        self.item_bias = self.param('item_bias', nn.initializers.zeros, (self.x_dim,), self.dtype)

    def denoise_logits(self, inputs):
        xt, y, t = self._split(inputs)
        U, V = self.U, self.V
        if self.freeze_lowrank:
            U, V = jax.lax.stop_gradient(U), jax.lax.stop_gradient(V)
        linear = self.ease_scale * ((y @ U) @ V.T) + self.obs_logit * y + self.item_bias
        return linear + self.core(jnp.concatenate([t * xt, l2_rows(y), t], axis=-1))


# registry: fmbayes names plus the CF extensions
MODEL_MAP = {**fm_models.MODEL_MAP, 'denoiser-gated': NNDenoiserGated, 'affine-tgated': NNAffineTGated,
             'gauss-ease': NNGaussEASE, 'denoiser-ease': NNDenoiserEASE}
RAW_Y_MODELS = {'gauss-ease', 'denoiser-ease'}       # models that take the binary observation (and normalise internally)


def build_model(velocity_param, x_dim, hidden_dim=600, depth=2, activation='swish', **kw):
    if velocity_param in fm_models.MODEL_MAP:
        return fm_models.build(velocity_param, x_dim, hidden_dim=hidden_dim, depth=depth,
                               activation=activation, **kw)
    return MODEL_MAP[velocity_param](hidden_dim=hidden_dim, x_dim=x_dim, depth=depth,
                                     activation=fm_models.ACTIVATIONS[activation], dtype=jnp.float32, **kw)


def denoiser_bce_loss(model):
    """`loss(params, x1, y, key)` for `NNDenoiserGated`: BCE of the data prediction D(x_t, y, t) against x1.

    Same sampling of (x0, t) as fmbayes' flow-matching loss; the minimiser is D = E[x1 | x_t, y], i.e.
    the flow-matching velocity through v = (D - x_t)/(1 - t). Summed over items, averaged over the batch.
    """
    logits_fn = jax.vmap(lambda p, u: model.apply(p, u, method=model.denoise_logits), in_axes=(None, 0))

    @jax.jit
    def loss(params, x1, y, key):
        x0key, tkey = jax.random.split(key)
        n, d = x1.shape
        x0 = draw_reference_jax(shape=(n, d), key=x0key, dtype=x1.dtype)
        t = jax.random.uniform(tkey, shape=(n, 1), dtype=x1.dtype)
        xt = t * x1 + (1 - t) * x0
        logits = logits_fn(params, jnp.hstack([xt, y, t]))
        bce = jnp.maximum(logits, 0) - logits * x1 + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.mean(jnp.sum(bce, axis=1))
    return loss


# ------------------------------------------------------------------------------------------------
# data: the forward operator and the fmbayes split
# ------------------------------------------------------------------------------------------------

def l2_rows(y):
    return y / jnp.maximum(jnp.linalg.norm(y, axis=-1, keepdims=True), 1e-8)


def make_y_fn(keep_prob: float, flip_prob: float = 0., noise_std: float = 0., normalise: bool = True) -> Callable:
    """Forward operator for `fit(y_fn=...)`: keep each positive w.p. `keep_prob`, add false positives
    (each non-interacted item w.p. `flip_prob`), l2-normalise the row, add N(0, noise_std^2) noise."""
    def y_fn(key, x):
        kk, fk, nk = jax.random.split(key, 3)
        y = x * jax.random.bernoulli(kk, keep_prob, x.shape).astype(x.dtype)
        if flip_prob > 0:
            y = jnp.maximum(y, (1 - x) * jax.random.bernoulli(fk, flip_prob, x.shape).astype(x.dtype))
        if normalise:
            y = l2_rows(y)
        if noise_std > 0:
            y = y + noise_std * jax.random.normal(nk, y.shape, y.dtype)
        return y
    return y_fn


@dataclass
class Split:
    """The fields `fmbayes.flows.train.fit` reads (mirrors `fmbayes.problems.eit.data.Split`)."""
    xg: object
    yg: object
    xv: object
    yv: object
    norm: dict = field(default_factory=dict)

    @property
    def dim(self):
        return self.xg.shape[1]

    @property
    def ydim(self):
        return self.yg.shape[1]


def _dense(X: sp.csr_matrix):
    return jnp.asarray(X.toarray(), dtype=jnp.float32)


def ease_params0(params, X: sp.csr_matrix, rank: int, keep_prob: float, edlae_params: dict,
                 cache: str | None = None, s2_init: str = 'popularity'):
    """Initial parameters for `NNGaussEASE` / `NNDenoiserEASE` (item_bias = logit of the popularity): U, V from the rank-`rank` eigen-approximation of EDLAE
    (`LowRankFactorization(method='eig')`), s^2 from item popularity, mean_scale = 1 / keep_prob (the
    observation keeps only that fraction of the items EDLAE was fitted on). `s2_init='one'` starts s^2 at 1,
    i.e. at fmbayes' affine skip a(t) x_t (no stiffness at t -> 1)."""
    from .linear import EDLAE
    from .lowrank import LowRankFactorization

    if cache and os.path.exists(cache):
        f = np.load(cache)
        U, V = f['U'], f['V']
    else:
        edlae = EDLAE(**edlae_params).fit(X)
        fac = LowRankFactorization(edlae.B, method='eig', G=edlae.regularized_gram(X)).truncate(rank)
        U, V = fac.U.astype(np.float32), fac.V.astype(np.float32)
        if cache:
            np.savez(cache, U=U, V=V)
    p = np.clip(np.asarray(X.mean(axis=0)).ravel(), 1e-6, 1 - 1e-6)
    new = dict(params['params'])
    new.update(U=jnp.asarray(U), V=jnp.asarray(V))
    if 'log_s2' in new:          # NNGaussEASE
        log_s2 = np.zeros_like(p) if s2_init == 'one' else np.log(np.clip(p * (1 - p), 1e-4, 0.25))
        new.update(log_s2=jnp.asarray(log_s2, jnp.float32), mean_scale=jnp.asarray(1.0 / keep_prob, jnp.float32))
    if 'item_bias' in new:       # NNDenoiserEASE: start at the item popularity
        new.update(item_bias=jnp.asarray(np.log(p / (1 - p)), jnp.float32))
    return {**params, 'params': new}


# ------------------------------------------------------------------------------------------------
# recommender
# ------------------------------------------------------------------------------------------------

class FMRecommender(Recommender):
    """
    velocity_param: fmbayes registry name ('plain', 'affine', ...) or a CF extension ('denoiser-gated')
    keep_prob:      forward operator used for training (0.8 = the test protocol)
    flip_prob:      observation noise: false positives added to y during training
    noise_std:      observation noise: additive Gaussian noise on the (l2-normalised) y during training;
                    at inference y is used noise-free
    select:         'last' (fmbayes default) or 'ndcg' (best validation NDCG@100, the report's protocol)
    score_samples:  0 = rank by the one-step posterior mean; S > 0 = by the average of S ODE samples
    """

    def __init__(self, velocity_param='denoiser-gated', hidden_dim=600, depth=2, activation='swish',
                 keep_prob=0.5, flip_prob=0., noise_std=0., epochs=30, batch_size=500, learning_rate=1e-3, optimizer='adamw',
                 weight_decay=0.1, schedule='cosine', select='ndcg', eval_every=1, steps=10, seed=0,
                 score_batch=1000, score_samples=0, rank=1000, edlae_params=None, freeze_lowrank=False,
                 s2_init='popularity', verbose=True):
        self.velocity_param = velocity_param
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.activation = activation
        self.keep_prob = keep_prob
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.schedule = schedule
        self.select = select
        self.eval_every = eval_every
        self.steps = steps
        self.seed = seed
        self.score_batch = score_batch
        self.score_samples = score_samples
        self.rank = rank
        self.freeze_lowrank = freeze_lowrank
        self.s2_init = s2_init
        self.edlae_params = edlae_params or {'lmbda': 300, 'p': 0.33}
        self.verbose = verbose
        self.normalise_y = velocity_param not in RAW_Y_MODELS

    def build(self, n_items: int):
        kw = {}
        if self.velocity_param == 'gauss-ease':
            kw = {'rank': self.rank, 'freeze_lowrank': self.freeze_lowrank}
        elif self.velocity_param == 'denoiser-ease':
            kw = {'rank': self.rank}
        self.model = build_model(self.velocity_param, n_items, self.hidden_dim, self.depth, self.activation, **kw)
        self._apply = jax.jit(jax.vmap(self.model.apply, in_axes=(None, 0)))
        self._transport = make_flow_transport(self.model, method='rk1', density=False, batched=True,
                                              from_steps=True)
        return self

    def fit(self, X: sp.csr_matrix, X_val_in: sp.csr_matrix | None = None, X_val_out: sp.csr_matrix | None = None):
        from ..evaluation import evaluate

        self.build(X.shape[1])
        y_fn = make_y_fn(self.keep_prob, self.flip_prob, self.noise_std, self.normalise_y)
        xg = _dense(X)
        # validation pairs: full validation histories, masked once with a fixed key
        xv = _dense(X_val_in + X_val_out) if X_val_in is not None else xg[:self.batch_size]
        yv = y_fn(jax.random.key(self.seed + 1), xv)
        split = Split(xg, xg, xv, yv)

        self.history, best = [], {'ndcg': -np.inf, 'params': None, 'epoch': -1}
        start = time.perf_counter()

        def on_epoch(epoch, train_loss, val_loss, _best_val, params):
            entry = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
            if X_val_in is not None and (epoch % self.eval_every == 0 or epoch == self.epochs):
                self.params = params
                entry['val_ndcg@100'] = evaluate(self, X_val_in, X_val_out, metrics=('ndcg@100',),
                                                 batch_size=self.score_batch)['ndcg@100']
                if entry['val_ndcg@100'] > best['ndcg']:
                    best.update(ndcg=entry['val_ndcg@100'], params=params, epoch=epoch)
            self.history.append(entry)
            if self.verbose:
                ndcg = f" | val ndcg@100 {entry['val_ndcg@100']:.4f}" if 'val_ndcg@100' in entry else ''
                print(f"epoch {epoch} | loss {train_loss:.4f} / val {val_loss:.4f}{ndcg} | "
                      f"{time.perf_counter() - start:.0f}s", flush=True)

        loss_fn = denoiser_bce_loss(self.model) if isinstance(self.model, NNDenoiserGated) else None
        params0 = None
        if isinstance(self.model, (NNGaussEASE, NNDenoiserEASE)):
            params0 = self.model.init(jax.random.key(self.seed), jnp.zeros(2 * X.shape[1] + 1))
            cache = os.path.join('results', f'edlae_lowrank_{X.shape[0]}x{X.shape[1]}_r{self.rank}_'
                                 f"l{self.edlae_params['lmbda']}_p{self.edlae_params['p']}.npz")
            params0 = ease_params0(params0, X, self.rank, self.keep_prob, self.edlae_params, cache, self.s2_init)
            self.params = params0
            if X_val_in is not None and self.verbose:
                ndcg0 = evaluate(self, X_val_in, X_val_out, metrics=('ndcg@100',), batch_size=self.score_batch)
                print(f"initialisation (low-rank EDLAE skip): val ndcg@100 {ndcg0['ndcg@100']:.4f}", flush=True)
        result = fm_train.fit(self.model, split, self.epochs, batch_size=self.batch_size,
                              learning_rate=self.learning_rate, seed=self.seed, on_epoch=on_epoch,
                              y_fn=y_fn, loss_fn=loss_fn, select='last', optimizer=self.optimizer,
                              weight_decay=self.weight_decay, schedule=self.schedule, params0=params0)
        self.params = best['params'] if self.select == 'ndcg' and best['params'] is not None else result.params
        self.best_epoch = best['epoch'] if self.select == 'ndcg' else result.epoch
        self.fit_time = time.perf_counter() - start
        del xg, xv, yv, split
        return self

    # ---- inference -------------------------------------------------------------------------
    def _cond(self, X: sp.csr_matrix):
        y = _dense(X)
        return l2_rows(y) if self.normalise_y else y

    def posterior_mean(self, X: sp.csr_matrix, key=None) -> np.ndarray:
        """x0 + v(0, x0; y) = E[x1 | y] (exact for the ideal field, any x0)."""
        y = self._cond(X)
        key = jax.random.key(self.seed) if key is None else key
        x0 = jax.random.normal(key, y.shape, dtype=y.dtype)
        t = jnp.zeros((y.shape[0], 1), y.dtype)
        return np.asarray(x0 + self._apply(self.params, jnp.hstack([x0, y, t])))

    def sample(self, X: sp.csr_matrix, n_samples: int, steps: int | None = None, key=None) -> np.ndarray:
        """Posterior samples by Euler transport, shape (n_samples, n_users, n_items)."""
        y = self._cond(X)
        key = jax.random.key(self.seed) if key is None else key
        yy = jnp.tile(y, (n_samples, 1))
        x0 = jax.random.normal(key, yy.shape, dtype=y.dtype)
        x1 = self._transport(self.params, x0, yy, steps or self.steps)
        return np.asarray(x1).reshape(n_samples, *y.shape)

    def score(self, X: sp.csr_matrix) -> np.ndarray:
        if self.score_samples:   # Monte Carlo posterior mean; batches keep samples x users x items on the device
            bs = max(1, self.score_batch // self.score_samples)
            return np.concatenate([self.sample(X[i:i + bs], self.score_samples).mean(0)
                                   for i in range(0, X.shape[0], bs)])
        return np.concatenate([self.posterior_mean(X[i:i + self.score_batch])
                               for i in range(0, X.shape[0], self.score_batch)])

    # ---- checkpoints (as in fmbayes: a pickle with params and config) ------------------------
    def get_config(self) -> dict:
        names = [n for n in inspect.signature(type(self).__init__).parameters if n != 'self']
        return {n: getattr(self, n) for n in names}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'params': jax.device_get(self.params), 'config': self.get_config(),
                         'n_items': self.model.x_dim, 'history': getattr(self, 'history', [])}, f)
        return path

    @classmethod
    def load(cls, path: str, **overrides) -> 'FMRecommender':
        with open(path, 'rb') as f:
            ckpt = pickle.load(f)
        model = cls(**{**ckpt['config'], **overrides}).build(ckpt['n_items'])
        model.params = jax.device_put(ckpt['params'])
        model.history = ckpt.get('history', [])
        return model
