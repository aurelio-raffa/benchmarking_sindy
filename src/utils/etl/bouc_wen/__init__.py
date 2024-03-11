import os
import scipy.io

import pandas as pd
import numpy as np


from src.utils import root_path


def load_data(
        training_path: str = 'data/BoucWen/Train signals/train_data.csv',
        test_path: str = 'data/BoucWen/Test signals/Validation signals',
        test_1_suffix: str = '_multisine',
        test_2_suffix: str = '_sinesweep',
        input_prefix: str = 'uval',
        output_prefix: str = 'yval',
        input_name: str = 'u',
        output_name: str = 'y',
        test_extension: str = '.mat',
        make_validation: bool = True,
        training_samples: int = 1000,
        validation_samples: int = 1000,
):
    # training and validation
    data = pd.read_csv(os.path.join(root_path, training_path))
    if make_validation:
        training_data = data.iloc[:training_samples, :].copy()
        validation_data = data.iloc[training_samples:training_samples + validation_samples, :].copy()
    else:
        training_data = data
        validation_data = None

    # test
    test_data = []
    for test_set in [test_1_suffix, test_2_suffix]:
        sys_out = scipy.io.loadmat(
            os.path.join(root_path, test_path, output_prefix + test_set + test_extension)
        )[output_prefix + test_set].T
        sys_inp = scipy.io.loadmat(
            os.path.join(root_path, test_path, input_prefix + test_set + test_extension)
        )[input_prefix + test_set].T

        test_data.append(
            pd.DataFrame(
                np.concatenate([sys_out, sys_inp], axis=1),
                columns=[output_name, input_name]
            )
        )

    return training_data, validation_data, *test_data
