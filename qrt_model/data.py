"""Loading of the challenge data into a (stock, date) design matrix."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

N_LAGS = 250
N_FACTORS = 10


def load_data(path="data"):
    """Load the training data and build the design matrix.

    Returns
    -------
    X : array (n_stocks * n_dates, 250)
        Lagged returns; column k is lag k + 1.
    y : array (n_stocks * n_dates,)
        Target returns.
    stock_id, date : arrays (n_stocks * n_dates,)
        Stock and target date of each row.

    Rows are stock-major: the n_dates rows of a stock are consecutive.
    """
    path = Path(path)
    X_train = pd.read_csv(path / "X_train.csv", index_col=0)
    Y_train = pd.read_csv(path / "Y_train.csv", index_col=0)
    X_train.columns = X_train.columns.astype(int)
    Y_train.columns = Y_train.columns.astype(int)
    if not (X_train.index == Y_train.index).all():
        raise ValueError("X_train and Y_train must list the stocks in the same order")

    X = sliding_window_view(X_train.to_numpy(), N_LAGS, axis=1)
    X = X[:, :-1, ::-1].reshape(-1, N_LAGS)
    y = Y_train.to_numpy().reshape(-1)

    n_stocks, n_dates = Y_train.shape
    stock_id = np.repeat(Y_train.index.to_numpy(), n_dates)
    date = np.tile(Y_train.columns.to_numpy(), n_stocks)
    assert X.shape == (n_stocks * n_dates, N_LAGS)
    return X, y, stock_id, date
