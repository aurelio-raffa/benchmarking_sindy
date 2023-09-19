import pandas as pd
import numpy as np


from __init__ import *


def import_cascaded_tanks_data(
        input_path: str = 'data/CascadedTanksFiles/dataBenchmark.csv',
):
    """Imports the cascaded tanks data.
    """
    # loads the raw files
    data = pd.read_csv(os.path.join(root_path, input_path))

    # extracts time information
    tf = data["Ts"].values[0]
    n = data.shape[0]
    t = np.linspace(0.0, tf, num=n)
    dt = tf / n

    # separates training and test signal
    train_data = data.loc[:, ['uEst', 'yEst']].copy().rename(columns={'uEst': 'u', 'yEst': 'y'})
    test_data = data.loc[:, ['uVal', 'yVal']].copy().rename(columns={'uVal': 'u', 'yVal': 'y'})

    return train_data, test_data, t, dt
