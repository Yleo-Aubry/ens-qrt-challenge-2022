"""Challenge metric and Stiefel constraint."""

import numpy as np

N_DATES = 504


def score(y_true, y_pred, n_dates=N_DATES):
    """Cosine similarity between predicted and true returns across stocks, averaged over dates.

    Both arrays must be stock-major (n_dates consecutive rows per stock).
    """
    Y = y_true.reshape(-1, n_dates)
    P = y_pred.reshape(-1, n_dates)
    num = (Y * P).sum(axis=0)
    den = np.linalg.norm(Y, axis=0) * np.linalg.norm(P, axis=0)
    return (num / den).mean()


def check_stiefel(A, tol=1e-6):
    """True if A.T @ A equals the identity up to tol on every entry."""
    return np.abs(A.T @ A - np.eye(A.shape[1])).max() < tol
