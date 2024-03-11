import os

import numpy as np
import pandas as pd

from src.utils.plotting.trajectory_plot import trajectory_plot
from src.utils.plotting import set_font


def plot_top_models(
        simulation_results: pd.DataFrame,
        simulation_scores: dict,
        top_n: int = 3
):
    scores_df = pd.DataFrame(simulation_scores).T.sort_values(by='rmse', ascending=True).copy()
    top_n_ids = scores_df.iloc[:top_n, :]['id'].values
    labels = scores_df.iloc[:top_n, :].index.tolist()

    t = simulation_results['t'].values
    u = simulation_results['u'].values

    xs = [np.nan * np.ones_like(u)] + [simulation_results[f'z.{model_id}'].values for model_id in top_n_ids]
    ys = [simulation_results['y']] + [simulation_results[f'y.{model_id}'].values for model_id in top_n_ids]

    set_font(publish=bool(os.getenv('PUBLISH')))

    return trajectory_plot(
        t=t,
        xs=ys,
        labels=['True'] + labels,
        x_subplot_label=None,
        x_scale='Output Voltage [V]',
    )
