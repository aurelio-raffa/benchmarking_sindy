import os
import pandas as pd

from __init__ import root_path


def load_benchmarks(benchmarks_path: str = 'data/Benchmarks/cascaded_tanks/cascaded_tanks.csv'):
    benchmark_data = pd.read_csv(os.path.join(root_path, benchmarks_path))

    return benchmark_data
