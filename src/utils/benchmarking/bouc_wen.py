import os
import pandas as pd

from __init__ import root_path


def load_benchmarks(
        benchmarks_path: str = 'data/Benchmarks/bouc_wen',
        lowfrequency: str = 'NARX_lowfrequency.csv',
        multisine: str = 'NARX_multisine.csv',
        sinesweep: str = 'NARX_sinesweep.csv'
):
    benchmark_lowfrqncy = pd.read_csv(os.path.join(root_path, benchmarks_path, lowfrequency)).to_numpy().reshape(-1, 1)
    benchmark_multisine = pd.read_csv(os.path.join(root_path, benchmarks_path, multisine)).to_numpy().reshape(-1, 1)
    benchmark_sinesweep = pd.read_csv(os.path.join(root_path, benchmarks_path, sinesweep)).to_numpy().reshape(-1, 1)

    return {'lowfrequency': benchmark_lowfrqncy, 'multisine': benchmark_multisine, 'sinesweep': benchmark_sinesweep}
