import os
import pandas as pd

from __init__ import root_path


def load_benchmarks(benchmarks_path: str = 'data/Benchmarks/pick_and_place/NARX.csv'):
    benchmark_data = pd.read_csv(os.path.join(root_path, benchmarks_path)).to_numpy().reshape(-1, 1)

    return benchmark_data
