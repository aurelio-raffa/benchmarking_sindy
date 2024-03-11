import os

import numpy as np
import pandas as pd
from mat73 import loadmat

from src.utils import root_path
from src.utils.preprocessing.compute_derivatives import compute_derivatives


def prepare_data(
        df: pd.DataFrame,
        dt: float,
        tvr_gamma: float = 10.0,
        derivation_order: int = 1
):
    u = df['u'].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)

    derivatives = compute_derivatives(y.ravel(), dt=dt, tvr_gamma=tvr_gamma, order=derivation_order)
    derivatives = [d.reshape(-1, 1) for d in derivatives]
    return y, u, *derivatives


def load_mat_data(p: str, key: str):
    return np.array(loadmat(os.path.join(root_path, p))[key])
