"""Model: blend of a date-weighted ridge and a smoothed direct cosine maximization.

Any submission (A, beta) predicts X @ A @ beta, so the model is fully described by the
vector w = A @ beta of 250 weights. The model estimates w, then writes it as A @ beta
with A on the Stiefel manifold.
"""

import numpy as np
from scipy.optimize import minimize

from .data import N_FACTORS
from .metrics import N_DATES

SEED = 42
RIDGE_LAMBDA = 1e4
SMOOTHING = 3e-5


def date_weights(y, p=1.0, n_dates=N_DATES):
    """Weight of each date, proportional to 1 / |y_t|^p (norm across stocks), mean 1."""
    n_stocks = len(y) // n_dates
    sw = np.linalg.norm(y.reshape(n_stocks, n_dates), axis=0) ** (-p)
    return sw / sw.mean()


def ridge_w(X, y, lam=RIDGE_LAMBDA, p=1.0, n_dates=N_DATES):
    """Ridge weights (250,), each date weighted by 1 / |y_t|^p. p=0 is the plain ridge."""
    n_stocks = len(y) // n_dates
    sw = np.tile(date_weights(y, p, n_dates), n_stocks)
    Xw = X * sw[:, None]
    return np.linalg.solve(Xw.T @ X + lam * np.eye(X.shape[1]), Xw.T @ y)


def cos_objective(v, Xr, Y, smoothing=0.0):
    """Objective minimized by cos_w, and its gradient with respect to v.

    -mean over dates of cos(Xr[:, t] @ v, Y[:, t]) + smoothing * 250 * sum_k (u_{k+1} - u_k)^2,
    with u = v / |v|. Xr has shape (n_stocks, n_dates, 250), Y (n_stocks, n_dates).
    """
    P = Xr @ v
    nP = np.linalg.norm(P, axis=0)
    nY = np.linalg.norm(Y, axis=0)
    c = (Y * P).sum(0) / (nY * nP)
    G = Y / (nY * nP) - c * P / nP**2
    f = -c.mean()
    g = -np.einsum("snk,sn->k", Xr, G) / Y.shape[1]
    if smoothing:
        nv = np.linalg.norm(v)
        u = v / nv
        du = np.diff(u)
        f += smoothing * len(v) * (du**2).sum()
        gu = 2 * smoothing * len(v) * (np.concatenate([[0], du]) - np.concatenate([du, [0]]))
        g += (gu - u * (u @ gu)) / nv
    return f, g


def cos_w(X, y, smoothing=SMOOTHING, w0=None, maxiter=1000, n_dates=N_DATES):
    """Unit-norm weights (250,) maximizing the mean cosine minus the smoothing penalty.

    L-BFGS started from the date-weighted ridge (or from w0).
    """
    n_stocks = len(y) // n_dates
    Xr = X.reshape(n_stocks, n_dates, -1)
    Y = y.reshape(n_stocks, n_dates)
    if w0 is None:
        w0 = ridge_w(X, y, n_dates=n_dates)
    w0 = w0 / np.linalg.norm(w0)
    res = minimize(cos_objective, w0, args=(Xr, Y, smoothing), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x / np.linalg.norm(res.x)


def blend_w(X, y, n_dates=N_DATES):
    """Model weights: sum of the normalized ridge and smoothed-cosine weights."""
    w_ridge = ridge_w(X, y, n_dates=n_dates)
    w_ridge /= np.linalg.norm(w_ridge)
    return w_ridge + cos_w(X, y, w0=w_ridge, n_dates=n_dates)


def complete_stiefel(w, seed=SEED):
    """A (250, 10) on the Stiefel manifold and beta (10,) such that A @ beta == w.

    The first column of A is w / |w|; the other 9 are random orthonormal columns that
    do not affect the prediction.
    """
    rng = np.random.default_rng(seed)
    A, _ = np.linalg.qr(np.column_stack([w, rng.standard_normal((len(w), N_FACTORS - 1))]))
    return A, A.T @ w


def fit(X, y, n_dates=N_DATES):
    """Fit the model, returns (A, beta)."""
    return complete_stiefel(blend_w(X, y, n_dates))


def fit_ridge(X, y, n_dates=N_DATES):
    """Reference model (date-weighted ridge alone), returns (A, beta)."""
    return complete_stiefel(ridge_w(X, y, n_dates=n_dates))
