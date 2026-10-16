"""Writing and reading of submission files."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import N_FACTORS, N_LAGS
from .metrics import check_stiefel


def read_submission(path):
    """Read a submission file, returns (A, beta)."""
    v = pd.read_csv(path, index_col=0).to_numpy().reshape(-1)
    return v[:-N_FACTORS].reshape(N_LAGS, N_FACTORS), v[-N_FACTORS:]


def save_submission(A, beta, path):
    """Write A (flattened row-major) then beta to a CSV file, and check the written file.

    Refuses to overwrite an existing file and to write an A that is not on the Stiefel
    manifold (the challenge scores such a submission -1).
    """
    path = Path(path)
    if A.shape != (N_LAGS, N_FACTORS) or beta.shape != (N_FACTORS,):
        raise ValueError(f"expected A {(N_LAGS, N_FACTORS)} and beta {(N_FACTORS,)}")
    if not check_stiefel(A):
        raise ValueError("A is not on the Stiefel manifold")
    if path.exists():
        raise FileExistsError(f"{path} already exists")

    pd.DataFrame(np.concatenate([A.reshape(-1), beta])).to_csv(path, lineterminator="\n")

    A_back, beta_back = read_submission(path)
    assert np.allclose(A_back, A) and np.allclose(beta_back, beta)
    assert check_stiefel(A_back)
