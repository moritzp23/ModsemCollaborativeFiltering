"""RecVAE (Shenbin et al., WSDM 2020). Network code from https://github.com/ilya-shenbin/RecVAE,
training loop from the seminar notebook (alternating encoder / decoder updates, model selection on
validation NDCG@100).
"""
from __future__ import annotations

import time
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from .base import Recommender


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(input_dim, hidden_dim)] +
                                [nn.Linear(hidden_dim, hidden_dim) for _ in range(4)])
        self.ln = nn.ModuleList([nn.LayerNorm(hidden_dim, eps=eps) for _ in range(5)])
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_rate):
        x = x / x.pow(2).sum(dim=-1).sqrt()[:, None]
        x = F.dropout(x, p=dropout_rate, training=self.training)
        # densely connected residual layers: h_k = LN(swish(fc_k(h_{k-1}) + h_1 + ... + h_{k-1}))
        hs = [self.ln[0](swish(self.fc[0](x)))]
        for fc, ln in zip(self.fc[1:], self.ln[1:]):
            hs.append(ln(swish(fc(hs[-1]) + sum(hs))))
        return self.fc_mu(hs[-1]), self.fc_logvar(hs[-1])


class CompositePrior(nn.Module):
    """Mixture of N(0, I), the posterior of the previous encoder, and a wide Gaussian."""

    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=(3 / 20, 3 / 4, 1 / 10)):
        super().__init__()
        self.mixture_weights = mixture_weights
        self.mu_prior = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)
        self.logvar_prior = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior = nn.Parameter(torch.full((1, latent_dim), 10.), requires_grad=False)
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        gaussians = [log_norm_pdf(z, self.mu_prior, self.logvar_prior),
                     log_norm_pdf(z, post_mu, post_logvar),
                     log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        return torch.logsumexp(torch.stack(gaussians, dim=-1), dim=-1)


class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super().__init__()
        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, user_ratings, beta=None, gamma=1., dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        if not calculate_loss:
            return x_pred
        # KL weight: gamma * |user history| (RecVAE) or constant beta (Mult-VAE)
        kl_weight = gamma * user_ratings.sum(dim=-1) if gamma else beta
        mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
        kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
        return (mll, kld), -(mll - kld)

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))


def _batches(X: sp.csr_matrix, batch_size: int, device, shuffle=False, rng=None):
    idx = rng.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
    for start in range(0, len(idx), batch_size):
        yield torch.as_tensor(X[idx[start:start + batch_size]].toarray(), device=device)


class RecVAE(Recommender):

    def __init__(self, hidden_dim=600, latent_dim=200, gamma=0.005, beta=None, lr=5e-4, batch_size=500,
                 n_epochs=50, n_enc_epochs=3, n_dec_epochs=1, dropout_rate=0.5, seed=1337,
                 device='cuda' if torch.cuda.is_available() else 'cpu', verbose=True):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.beta = beta
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_enc_epochs = n_enc_epochs
        self.n_dec_epochs = n_dec_epochs
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.device = torch.device(device)
        self.verbose = verbose

    def _train(self, X, optimizer, n_epochs, dropout_rate):
        self.net.train()
        for _ in range(n_epochs):
            for batch in _batches(X, self.batch_size, self.device, shuffle=True, rng=self._rng):
                optimizer.zero_grad()
                _, loss = self.net(batch, beta=self.beta, gamma=self.gamma, dropout_rate=dropout_rate)
                loss.backward()
                optimizer.step()

    def fit(self, X: sp.csr_matrix, X_val_in: sp.csr_matrix | None = None, X_val_out: sp.csr_matrix | None = None):
        """Alternating training; if validation data is given, keep the epoch with the best NDCG@100."""
        from ..evaluation import evaluate

        torch.manual_seed(self.seed)
        self._rng = np.random.default_rng(self.seed)
        self.net = VAE(self.hidden_dim, self.latent_dim, X.shape[1]).to(self.device)
        opt_enc = torch.optim.Adam(self.net.encoder.parameters(), lr=self.lr)
        opt_dec = torch.optim.Adam(self.net.decoder.parameters(), lr=self.lr)

        best_ndcg, best_state, self.history = -np.inf, None, []
        start = time.perf_counter()
        for epoch in range(self.n_epochs):
            self._train(X, opt_enc, self.n_enc_epochs, self.dropout_rate)
            self.net.update_prior()
            self._train(X, opt_dec, self.n_dec_epochs, 0.)
            if X_val_in is not None:
                ndcg = evaluate(self, X_val_in, X_val_out, metrics=('ndcg@100',))['ndcg@100']
                self.history.append(ndcg)
                if ndcg > best_ndcg:
                    best_ndcg, best_state = ndcg, deepcopy(self.net.state_dict())
                if self.verbose:
                    print(f'epoch {epoch} | valid ndcg@100: {ndcg:.4f} | best: {best_ndcg:.4f} | '
                          f'{time.perf_counter() - start:.0f}s')
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fit_time = time.perf_counter() - start
        return self

    @torch.no_grad()
    def score(self, X: sp.csr_matrix) -> np.ndarray:
        self.net.eval()
        return torch.cat([self.net(b, calculate_loss=False).cpu()
                          for b in _batches(X, self.batch_size, self.device)]).numpy()
