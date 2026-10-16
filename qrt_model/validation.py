"""Cross-validation across stocks.

The hidden test set contains other stocks over the same dates, so the folds split the
stocks, never the dates.
"""

import numpy as np

from .metrics import score

SEEDS = (42, 43, 44, 45, 46)


def stock_folds(stock_id, n_folds=5, seed=SEEDS[0]):
    """Random partition of the stocks into n_folds groups (list of arrays of stock ids)."""
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(np.unique(stock_id)), n_folds)


def cv_score(fit, X, y, stock_id, n_folds=5, seed=SEEDS[0], **fit_kwargs):
    """Score of fit on each fold of held-out stocks.

    fit(X, y, **fit_kwargs) must return (A, beta). Subsets are taken with boolean masks,
    which keeps the stock-major order required by score.
    """
    scores = []
    for test_stocks in stock_folds(stock_id, n_folds, seed):
        test = np.isin(stock_id, test_stocks)
        A, beta = fit(X[~test], y[~test], **fit_kwargs)
        scores.append(score(y[test], X[test] @ A @ beta))
    return np.array(scores)


def cv_repeated(fit, X, y, stock_id, seeds=SEEDS, n_folds=5, **fit_kwargs):
    """cv_score repeated over several seeds; returns all fold scores concatenated.

    The folds depend only on the seeds, so two models evaluated with the same seeds can
    be compared fold by fold (see summary).
    """
    return np.concatenate([cv_score(fit, X, y, stock_id, n_folds, s, **fit_kwargs) for s in seeds])


def summary(s, s_ref=None, n_folds=5):
    """Mean and standard error of fold scores s; with s_ref, also the paired gain s - s_ref.

    The repeated splits reuse the same stocks, so the folds are not independent. The
    standard error divides the fold standard deviation by sqrt(n_folds), not by
    sqrt(len(s)).
    """
    out = {"cv": s.mean(), "se": s.std() / np.sqrt(n_folds)}
    if s_ref is not None:
        d = s - s_ref
        out["gain"] = d.mean()
        out["se_gain"] = d.std() / np.sqrt(n_folds)
    return out
