from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrt_model import (blend_w, check_stiefel, complete_stiefel, cos_objective, cos_w, cv_repeated,
                       cv_score, date_weights, fit, load_data, read_submission, ridge_w,
                       save_submission, score, stock_folds, summary)

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)


def toy_data(n_stocks=5, n_dates=504):
    stock_id = np.repeat(np.arange(n_stocks), n_dates)
    X = rng.standard_normal((len(stock_id), 250))
    y = X[:, :3].sum(1) + 3 * rng.standard_normal(len(stock_id))
    return X, y, stock_id


def test_score_toy():
    y_true = np.array([1, 2, 3, 4, 5, 6.])
    y_pred = np.array([1, 0, 3, 4, 1, 0.])
    assert np.isclose(score(y_true, y_pred, n_dates=3), 0.792, atol=1e-3)


def test_score_properties():
    y = rng.standard_normal(50 * 504)
    p = rng.standard_normal(len(y))
    assert np.isclose(score(y, y), 1)
    assert np.isclose(score(y, -y), -1)
    assert np.isclose(score(y, p), score(y, 3 * p))


def test_check_stiefel():
    Q = np.linalg.qr(rng.standard_normal((250, 10)))[0]
    assert check_stiefel(np.eye(250)[:, :10])
    assert check_stiefel(Q)
    assert not check_stiefel(2 * Q)


def test_complete_stiefel():
    w = rng.standard_normal(250)
    A, beta = complete_stiefel(w)
    assert A.shape == (250, 10) and beta.shape == (10,)
    assert check_stiefel(A)
    assert np.allclose(A @ beta, w)


def test_ridge_closed_form():
    X, y, _ = toy_data()
    assert np.allclose(ridge_w(X, y, p=0), np.linalg.solve(X.T @ X + 1e4 * np.eye(250), X.T @ y))
    # constant date norms: the date weights are all 1 and p has no effect
    y_const = np.tile(np.where(np.arange(504) % 2, 1., -1.), 5)
    assert np.allclose(date_weights(y_const), 1)
    assert np.allclose(ridge_w(X, y_const, p=1), ridge_w(X, y_const, p=0))


@pytest.mark.parametrize("smoothing", [0.0, 1e-2])
def test_cos_objective_gradient(smoothing):
    Xr = rng.standard_normal((6, 30, 250))
    Y = rng.standard_normal((6, 30))
    v, d = rng.standard_normal(250), rng.standard_normal(250)
    eps = 1e-6
    f_plus = cos_objective(v + eps * d, Xr, Y, smoothing)[0]
    f_minus = cos_objective(v - eps * d, Xr, Y, smoothing)[0]
    assert np.isclose((f_plus - f_minus) / (2 * eps), cos_objective(v, Xr, Y, smoothing)[1] @ d, rtol=1e-5)
    assert np.isclose(cos_objective(v, Xr, Y)[0], -score(Y.reshape(-1), Xr.reshape(-1, 250) @ v, n_dates=30))


def test_cos_w_beats_ridge_in_sample():
    X, y, _ = toy_data()
    w = cos_w(X, y, smoothing=0, maxiter=200)
    assert np.isclose(np.linalg.norm(w), 1)
    assert score(y, X @ w) >= score(y, X @ ridge_w(X, y)) - 1e-9


def test_fit_returns_blend_on_stiefel():
    X, y, _ = toy_data()
    A, beta = fit(X, y)
    assert check_stiefel(A)
    assert np.allclose(A @ beta, blend_w(X, y))


def test_cv_score_never_fits_on_test_stocks():
    stock_id = np.repeat(np.arange(50), 504)
    X = rng.standard_normal((len(stock_id), 250))
    y = rng.standard_normal(len(stock_id))
    seen = []

    def fit_stub(X_fit, y_fit):
        seen.append(len(y_fit))
        return np.eye(250)[:, :10], np.ones(10)

    assert cv_score(fit_stub, X, y, stock_id).shape == (5,)
    assert seen == [40 * 504] * 5
    folds = stock_folds(stock_id)
    assert sorted(np.concatenate(folds)) == list(range(50))


def test_cv_repeated_and_summary():
    stock_id = np.repeat(np.arange(50), 504)
    X = rng.standard_normal((len(stock_id), 250))
    y = X[:, 0] + rng.standard_normal(len(stock_id))

    def fit_stub(X_fit, y_fit):
        return np.eye(250)[:, :10], np.ones(10)

    s = cv_repeated(fit_stub, X, y, stock_id, seeds=(1, 2))
    assert s.shape == (10,)
    assert np.allclose(s[5:], cv_score(fit_stub, X, y, stock_id, seed=2))
    r = summary(s, s)
    assert r["gain"] == 0 and r["se_gain"] == 0
    assert np.isclose(r["se"], s.std() / np.sqrt(5))


def test_save_submission(tmp_path):
    A = np.linalg.qr(rng.standard_normal((250, 10)))[0]
    beta = rng.standard_normal(10)
    out = tmp_path / "submission.csv"
    save_submission(A, beta, out)
    assert pd.read_csv(out, index_col=0).shape == (2510, 1)
    A_back, beta_back = read_submission(out)
    assert np.allclose(A_back, A) and np.allclose(beta_back, beta)
    with pytest.raises(FileExistsError):
        save_submission(A, beta, out)
    with pytest.raises(ValueError):
        save_submission(2 * A, beta, tmp_path / "other.csv")


def test_reproduces_submitted_file():
    X, y, _, _ = load_data(ROOT / "data")
    A, beta = fit(X, y)
    A_sub, beta_sub = read_submission(ROOT / "soumission" / "soumission_modele4.csv")
    assert np.allclose(A, A_sub) and np.allclose(beta, beta_sub)
    assert np.isclose(score(y, X @ A @ beta), 0.1462, atol=1e-4)
