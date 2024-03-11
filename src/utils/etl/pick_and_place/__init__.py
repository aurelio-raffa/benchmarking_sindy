import os

import pandas as pd

from src.utils import root_path
from src.utils.etl import load_mat_data


def load_data(
        input_path: str = 'data/PickAndPlace/matlab_files',
        train_dir: str = 'train',
        test_dir: str = 'test',
        y_filename: str = 'y.mat',
        u_filename: str = 'u.mat',
        y_key_train: str = 'y_train',
        u_key_train: str = 'u_train',
        y_key_test: str = 'y_val',
        u_key_test: str = 'u_val',
        training_samples: int = 3840,
        validation_samples: int = 960
):
    # from Bemporad et al., 2018:
    # - training: 3840 samples
    # - validation: 960 samples
    # - test: 1200 samples

    # 01 - loading training
    y_train = load_mat_data(
        os.path.join(input_path, train_dir, y_filename),
        y_key_train
    )
    u_train = load_mat_data(
        os.path.join(input_path, train_dir, u_filename),
        u_key_train
    )

    # 02 - stashing validation
    y_val = y_train[training_samples:training_samples + validation_samples].copy()
    u_val = u_train[training_samples:training_samples + validation_samples].copy()
    y_train = y_train[:training_samples].copy()
    u_train = u_train[:training_samples].copy()

    # 03 - loading test
    y_test = load_mat_data(os.path.join(root_path, input_path, test_dir, y_filename), y_key_test)
    u_test = load_mat_data(os.path.join(root_path, input_path, test_dir, u_filename), u_key_test)

    # preparing the data in pandas format
    train_data = pd.DataFrame(
        data={
            'y': y_train.ravel(),
            'u': u_train.ravel()
        }
    )
    validation_data = pd.DataFrame(
        data={
            'y': y_val.ravel(),
            'u': u_val.ravel()
        }
    )
    test_data = pd.DataFrame(
        data={
            'y': y_test.ravel(),
            'u': u_test.ravel()
        }
    )

    return train_data, validation_data, test_data
