import os

import pandas as pd
import numpy as np


from src.utils import root_path


def load_data(
        input_path: str = 'data/CascadedTanksFiles/dataBenchmark.csv',
        make_validation: bool = True,
        training_samples: int = 768,
        validation_samples: int = 256
):
    """Imports the cascaded tanks data.
    """
    # loads the raw files
    data = pd.read_csv(os.path.join(root_path, input_path))

    # extracts time information
    tf = data["Ts"].values[0]
    n = data.shape[0]
    t_train = t_test = np.linspace(0.0, tf, num=n)
    dt = tf / n

    # separates training and test signal
    train_data = data.loc[:, ['uEst', 'yEst']].copy().rename(columns={'uEst': 'u', 'yEst': 'y'})
    test_data = data.loc[:, ['uVal', 'yVal']].copy().rename(columns={'uVal': 'u', 'yVal': 'y'})

    if make_validation:
        validation_data = train_data.iloc[training_samples:training_samples + validation_samples, :].copy()
        train_data = train_data.iloc[:training_samples, :].copy()
        t_validation = t_train[training_samples:training_samples + validation_samples].copy()
        t_validation -= t_validation[0]
        t_train = t_train[:training_samples].copy()
    else:
        validation_data = None
        t_validation = None

    return train_data, validation_data, test_data, dt, t_train, t_validation, t_test
