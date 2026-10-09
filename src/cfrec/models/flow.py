"""Amortized Bayesian conditional flow matching for collaborative filtering.

Inverse-problem view: the unknown is a user's full interaction vector x in {0,1}^I, the observation is
y = M(x), a random subset of the user's positives (forward operator: keep each positive with probability
`keep_prob`). Joint samples (x, y) are simulated from the training users. A conditional flow transports
N(0, I) to the posterior p(x | y) (Blechschmidt, Ernst, Poguntke, Sprungk 2026):

    x_t = (1 - t) x_0 + t x_1,   x_0 ~ N(0, I),  (x_1, y) ~ joint,
    v(t, x, y) = E[x_1 - x_0 | x_t = x, y] = (E[x_1 | x_t = x, y] - x) / (1 - t).

The network predicts the posterior "denoiser" x1_hat(t, x_t, y) = E[x_1 | x_t, y] (data prediction), which
avoids pushing the identity-like -x_0 component of the velocity through a low-dimensional bottleneck.

Since x_0 is independent of (x_1, y), x1_hat(0, x_0, y) = E[x_1 | y]: the posterior mean -- all that is
needed for ranking -- costs a single network evaluation. Posterior samples (for uncertainty
quantification) are obtained by integrating the ODE.
"""
from __future__ import annotations

import math
import time
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from .base import Recommender


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, n_freq: int = 32):
        super().__init__()
        self.register_buffer('freq', torch.exp(torch.linspace(0, math.log(1000.), n_freq)))
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        arg = t[:, None] * self.freq[None, :]
        return self.mlp(torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1))


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(dim, dim))

    def forward(self, h):
        return h + self.net(h)


class ConditionalDenoiser(nn.Module):
    """x1_hat(t, x_t, y): logits of E[x_1 | x_t, y] for each item."""

    def __init__(self, n_items: int, hidden_dim: int = 600, n_blocks: int = 2, input_dropout: float = 0.,
                 dropout: float = 0., x_gate: bool = True):
        super().__init__()
        self.x_gate = x_gate
        self.enc_y = nn.Linear(n_items, hidden_dim)   # observation encoder (Mult-VAE style, l2-normalized)
        self.enc_x = nn.Linear(n_items, hidden_dim)   # current state of the flow
        self.time = TimeEmbedding(hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_items))
        self.input_dropout = input_dropout

    def forward(self, t, x_t, y):
        y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        y = F.dropout(y, p=self.input_dropout, training=self.training)
        h_x = self.enc_x(x_t)
        if self.x_gate:
            # at t = 0, x_t = x_0 carries no information about x_1: the optimal output is independent of it
            h_x = t[:, None] * h_x
        h = torch.tanh(self.enc_y(y)) + h_x + self.time(t)
        return self.out(self.blocks(h))


def _dense_batches(X: sp.csr_matrix, batch_size: int, device, idx=None):
    idx = np.arange(X.shape[0]) if idx is None else idx
    for start in range(0, len(idx), batch_size):
        yield torch.as_tensor(X[idx[start:start + batch_size]].toarray(), device=device)


class FlowMatchingCF(Recommender):
    """
    keep_prob:      forward operator, each positive is observed with this probability (0.8 = test protocol)
    loss:           'bce' (default) or 'mse' on the data prediction; both are minimized by E[x_1 | x_t, y]
    t_sampling:     'uniform' or 'logit_normal' (more weight on intermediate t)
    score_mode:     'mean' = one-step posterior mean x1_hat(0, x_0, y);  'ode' = average of ODE samples
    """

    def __init__(self, hidden_dim=600, n_blocks=2, keep_prob=0.8, input_dropout=0., dropout=0., x_gate=True, loss='bce',
                 t_sampling='uniform', lr=1e-3, weight_decay=0., batch_size=500, n_epochs=50, eval_every=1,
                 score_mode='mean', n_samples=8, n_steps=10, seed=0,
                 device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.keep_prob = keep_prob
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.x_gate = x_gate
        self.loss = loss
        self.t_sampling = t_sampling
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eval_every = eval_every
        self.score_mode = score_mode
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.seed = seed
        self.device = torch.device(device)
        self.verbose = verbose

    # ------------------------------------------------------------------ training
    def _sample_t(self, n):
        if self.t_sampling == 'uniform':
            return torch.rand(n, device=self.device)
        if self.t_sampling == 'logit_normal':
            return torch.sigmoid(torch.randn(n, device=self.device))
        raise NotImplementedError(self.t_sampling)

    def _loss(self, x1):
        """Simulate the forward operator, draw (t, x_0) and regress the posterior denoiser."""
        y = x1 * (torch.rand_like(x1) < self.keep_prob)
        t = self._sample_t(x1.shape[0])
        x0 = torch.randn_like(x1)
        x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
        logits = self.net(t, x_t, y)
        if self.loss == 'bce':
            return F.binary_cross_entropy_with_logits(logits, x1, reduction='none').sum(-1).mean()
        if self.loss == 'mse':
            return (torch.sigmoid(logits) - x1).pow(2).sum(-1).mean()
        raise NotImplementedError(self.loss)

    def fit(self, X: sp.csr_matrix, X_val_in: sp.csr_matrix | None = None, X_val_out: sp.csr_matrix | None = None):
        from ..evaluation import evaluate

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_items = X.shape[1]
        self.net = ConditionalDenoiser(self.n_items, self.hidden_dim, self.n_blocks, self.input_dropout,
                                       self.dropout, self.x_gate).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best, best_state, self.history = -np.inf, None, []
        start = time.perf_counter()
        for epoch in range(self.n_epochs):
            self.net.train()
            losses = []
            for x1 in _dense_batches(X, self.batch_size, self.device, rng.permutation(X.shape[0])):
                opt.zero_grad()
                loss = self._loss(x1)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            entry = {'epoch': epoch, 'loss': float(np.mean(losses))}
            if X_val_in is not None and (epoch + 1) % self.eval_every == 0:
                entry['val_ndcg@100'] = evaluate(self, X_val_in, X_val_out, metrics=('ndcg@100',))['ndcg@100']
                if entry['val_ndcg@100'] > best:
                    best, best_state = entry['val_ndcg@100'], deepcopy(self.net.state_dict())
            self.history.append(entry)
            if self.verbose:
                val = f" | val ndcg@100 {entry['val_ndcg@100']:.4f} (best {best:.4f})" if 'val_ndcg@100' in entry else ''
                print(f"epoch {epoch} | loss {entry['loss']:.2f}{val} | {time.perf_counter() - start:.0f}s", flush=True)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fit_time = time.perf_counter() - start
        return self

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def posterior_mean(self, y: torch.Tensor) -> torch.Tensor:
        """E[x_1 | y] = x1_hat(0, x_0, y) for any x_0 (exact for the ideal denoiser); x_0 ~ N(0, I)."""
        self.net.eval()
        t = torch.zeros(y.shape[0], device=y.device)
        return torch.sigmoid(self.net(t, torch.randn_like(y), y))

    @torch.no_grad()
    def sample(self, y: torch.Tensor, n_samples: int | None = None, n_steps: int | None = None) -> torch.Tensor:
        """Posterior samples via Euler integration of dx/dt = (x1_hat - x) / (1 - t); shape (n_samples, n, I)."""
        self.net.eval()
        n_samples, n_steps = n_samples or self.n_samples, n_steps or self.n_steps
        yy = y.repeat(n_samples, 1)
        x = torch.randn_like(yy)
        ts = torch.linspace(0, 1, n_steps + 1, device=y.device)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            x1_hat = torch.sigmoid(self.net(t0.expand(len(x)), x, yy))
            x = x + (t1 - t0) * (x1_hat - x) / (1 - t0)
        return x.view(n_samples, *y.shape)

    @torch.no_grad()
    def score(self, X: sp.csr_matrix) -> np.ndarray:
        out = []
        for y in _dense_batches(X, self.batch_size, self.device):
            if self.score_mode == 'mean':
                out.append(self.posterior_mean(y).cpu())
            elif self.score_mode == 'ode':
                out.append(self.sample(y).mean(0).cpu())
            else:
                raise NotImplementedError(self.score_mode)
        return torch.cat(out).numpy()
