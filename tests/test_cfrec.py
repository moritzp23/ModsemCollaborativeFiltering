import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cfrec.data import to_csr
from cfrec.evaluation import evaluate
from cfrec.linalg import spd_inverse
from cfrec.metrics import hits, ndcg, recall, topk
from cfrec.models import ADMMSlim, DLAE, EASE, EDLAE, LowRankFactorization, MostPopular, MRFApprox, MRFDense

sys.path.insert(0, str(Path(__file__).parents[1] / 'legacy'))
import metrics as legacy_metrics  # noqa: E402


@pytest.fixture(scope='module')
def X():
    rng = np.random.default_rng(0)
    # popularity-skewed binary interactions, 300 users x 40 items
    p = np.clip(rng.gamma(0.8, 0.15, size=40), 0.01, 0.9)
    return sp.csr_matrix((rng.random((300, 40)) < p).astype(np.float32))


def test_metrics_match_legacy():
    rng = np.random.default_rng(1)
    n_users, n_items, k = 200, 300, 100
    X_true = sp.csr_matrix((rng.random((n_users, n_items)) < 0.05).astype(np.float32))
    X_true[np.diff(X_true.indptr) == 0, 0] = 1.  # every user needs a relevant item
    X_true = sp.csr_matrix(X_true)
    top = topk(rng.random((n_users, n_items)).astype(np.float32), k)
    hit, n_true = hits(top, X_true), np.diff(X_true.indptr)
    for kk in (20, 50, 100):
        legacy_r = [legacy_metrics.Recall(k=kk)(top[u], X_true[u].indices) for u in range(n_users)]
        legacy_n = [legacy_metrics.NDCG(k=kk)(top[u], X_true[u].indices) for u in range(n_users)]
        np.testing.assert_allclose(recall(hit, n_true, kk), legacy_r)
        np.testing.assert_allclose(ndcg(hit, n_true, kk), legacy_n, rtol=1e-9)


def test_topk_sorted():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]], dtype=np.float32)
    np.testing.assert_array_equal(topk(scores, 3), [[1, 3, 2]])


def test_spd_inverse():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(50, 50))
    A = A @ A.T + 50 * np.eye(50)
    np.testing.assert_allclose(spd_inverse(A.copy()), np.linalg.inv(A), atol=1e-10)
    np.testing.assert_allclose(spd_inverse(A.astype(np.float32)), np.linalg.inv(A), atol=1e-5)


def test_to_csr_aligns_rows():
    tr = pd.DataFrame({'uid': [5, 5, 7, 9], 'sid': [0, 1, 2, 3]})
    te = pd.DataFrame({'uid': [5, 9], 'sid': [2, 0]})
    users = np.union1d(tr.uid, te.uid)
    X_tr, X_te = to_csr(tr, 4, users), to_csr(te, 4, users)
    assert X_tr.shape == X_te.shape == (3, 4)
    assert X_te[1].nnz == 0  # user 7 has no held-out items -> skipped in evaluation
    assert list(X_te[2].indices) == [0]


def test_ease_kkt(X):
    """EASE solves min ||X - XB||^2 + lambda ||B||^2 s.t. diag(B)=0: gradient vanishes off the diagonal."""
    lmbda = 5.
    B = EASE(lmbda).fit(X).B.astype(np.float64)
    G = (X.T @ X).toarray().astype(np.float64)
    grad = G @ B - G + lmbda * B
    off_diag = ~np.eye(G.shape[0], dtype=bool)
    assert np.abs(np.diag(B)).max() == 0
    np.testing.assert_allclose(grad[off_diag], 0, atol=1e-2)


def test_dlae_chol_equals_inv(X):
    a = DLAE(10., 0.3, method='chol').fit(X).score(X[:20])
    b = DLAE(10., 0.3, method='inv').fit(X).score(X[:20])
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_mrf_full_graph_equals_dense(X):
    """With a complete graph, r=1 and alpha=0, the block approximation is the dense MRF solution."""
    dense = MRFDense(lmbda=5.).fit(X)
    approx = MRFApprox(lmbda=5., alpha=0., threshold=-1., max_in_col=1000, r=1.).fit(X)
    # item pairs with exactly zero empirical covariance are not edges of the graph
    XtX = (X.T @ X).toarray()
    mu = np.diag(XtX) / X.shape[0]
    edges = (XtX - np.outer(mu, mu * X.shape[0])) != 0
    np.testing.assert_allclose(approx.B.toarray()[edges], dense.B[edges], atol=1e-5)
    assert np.all(approx.B.toarray()[~edges] == 0)


def test_admm_without_l1_converges_to_edlae(X):
    edlae = EDLAE(lmbda=10., p=0.2).fit(X)
    admm = ADMMSlim(lambda1=0., lambda2=10., p=0.2, rho=50., n_iter=300).fit(X)
    np.testing.assert_allclose(admm.B.toarray(), edlae.B, atol=1e-4)


def test_lowrank_full_rank_recovers_B(X):
    model = EDLAE(lmbda=10., p=0.2).fit(X)
    for method in ('svd', 'eig'):
        G = model.regularized_gram(X) if method == 'eig' else None
        lr = LowRankFactorization(model.B, method=method, G=G).truncate(X.shape[1])
        np.testing.assert_allclose(lr.U @ lr.V.T, model.B, atol=1e-4)


def test_evaluate_excludes_seen_items(X):
    res = evaluate(MostPopular().fit(X), X[:50], X[50:100], metrics=('recall@5',), per_user=True)
    assert 0 <= res['recall@5'] <= 1


def test_recvae_smoke(X):
    pytest.importorskip('torch')
    from cfrec.models import RecVAE
    model = RecVAE(hidden_dim=16, latent_dim=4, n_epochs=2, batch_size=64, device='cpu', verbose=False)
    model.fit(X[:200], X[200:250], X[250:])
    assert model.score(X[:7]).shape == (7, X.shape[1])
    assert len(model.history) == 2


def test_calibration_platt_and_ece():
    from scipy.special import expit
    from cfrec.calibration import CalibrationAccumulator, fit_platt
    rng = np.random.default_rng(3)
    n_users, n_items = 400, 200
    X_in = sp.csr_matrix((rng.random((n_users, n_items)) < 0.02).astype(np.float32))
    scores = rng.normal(size=(n_users, n_items))
    probs = expit(1.5 * scores - 2.0)
    X_out = sp.csr_matrix(((rng.random((n_users, n_items)) < probs) & (X_in.toarray() == 0)).astype(np.float32))
    a, b = fit_platt(lambda X: scores[:X.shape[0]] if X.shape[0] == n_users else None, X_in, X_out,
                     batch_size=n_users)
    assert abs(a - 1.5) < 0.1 and abs(b + 2.0) < 0.1
    acc = CalibrationAccumulator(top_k=10)
    acc.update(probs, X_in, X_out)
    assert acc.summary()['ece_all'] < 0.01


@pytest.mark.parametrize('velocity_param', ['affine', 'affine-tgated', 'denoiser-gated', 'gauss-ease',
                                            'denoiser-ease'])
def test_fmbayes_models_smoke(X, velocity_param, tmp_path, monkeypatch):
    pytest.importorskip('fmbayes')
    import jax
    import jax.numpy as jnp
    from cfrec.models import FMRecommender
    monkeypatch.chdir(tmp_path)  # gauss-ease caches its EDLAE factors under ./results
    (tmp_path / 'results').mkdir()
    m = FMRecommender(velocity_param=velocity_param, hidden_dim=16, rank=8, epochs=1, batch_size=50,
                      edlae_params={'lmbda': 10., 'p': 0.2}, verbose=False)
    m.fit(X[:200], X[200:250], X[250:])
    assert m.score(X[:7]).shape == (7, X.shape[1])
    assert np.isfinite(m.sample(X[:3], n_samples=2, steps=4)).all()
    if velocity_param != 'affine':  # time-gated: one-step posterior mean independent of x0
        y = m._cond(X[:1])[0]
        x0 = jax.random.normal(jax.random.key(0), y.shape)
        f = lambda x: x + m.model.apply(m.params, jnp.hstack([x, y, 0.]))
        assert jnp.allclose(f(x0), f(-x0), atol=1e-4)


def test_fmbayes_save_load_roundtrip(X, tmp_path):
    pytest.importorskip('fmbayes')
    from cfrec.models import FMRecommender
    m = FMRecommender(velocity_param='denoiser-gated', hidden_dim=16, epochs=1, batch_size=50, verbose=False)
    m.fit(X[:200], X[200:250], X[250:])
    loaded = FMRecommender.load(m.save(str(tmp_path / 'fm.pkl')), verbose=False)
    np.testing.assert_allclose(loaded.score(X[:5]), m.score(X[:5]), rtol=1e-6)
    assert loaded.get_config() == {**m.get_config(), 'verbose': False}


def test_joint_scores_detect_dependence():
    """Items that always co-occur: samples with the right joint beat independent draws with the same marginals."""
    from cfrec.calibration import JointAccumulator
    rng = np.random.default_rng(4)
    n_users, n_items, m = 300, 20, 64
    X_in = sp.csr_matrix((n_users, n_items), dtype=np.float32)
    on = rng.random(n_users) < 0.5                     # all-or-nothing users
    X_out = sp.csr_matrix(np.repeat(on[:, None], n_items, axis=1).astype(np.float32))
    p = np.full((n_users, n_items), 0.5)
    joint_true = np.repeat((rng.random((m, n_users)) < 0.5)[:, :, None], n_items, axis=2)
    indep = rng.random((m, n_users, n_items)) < p
    acc_true, acc_indep = JointAccumulator(top_k=5), JointAccumulator(top_k=5)
    acc_true.update(joint_true, p, X_in, X_out)
    acc_indep.update(indep, p, X_in, X_out)
    s_true, s_indep = acc_true.summary(), acc_indep.summary()
    assert s_true['energy'] < s_indep['energy']
    assert s_true['crps_count'] < s_indep['crps_count']
