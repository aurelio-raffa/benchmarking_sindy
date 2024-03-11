import numpy as np
import pandas as pd

from src.utils.model_selection import score_simulation


def prepare_simulation_data(
        t_test,
        x_test,
        u_test,
        benchmark_data,
        baseline_models_names,
        baseline_model_types,
        baseline_simulations,
        baseline_scores,
        hidden_models_names,
        hidden_model_types,
        hidden_simulations,
        hidden_reconstructions,
        hidden_scores,
        rescale_factor: float = 10.0,
):
    simulation_cols = \
        ['t', 'u', 'y'] \
        + [f'y.baseline.{mtype}{i}' for i, mtype in enumerate(baseline_model_types)] \
        + [f'y.hidden.{mtype}{i}' for i, mtype in enumerate(hidden_model_types)]
    simulation_results = pd.DataFrame(
        np.concatenate(
            [t_test.reshape(-1, 1) / rescale_factor, u_test, x_test] + baseline_simulations + hidden_simulations,
            axis=1
        ) * rescale_factor,
        columns=simulation_cols
    )
    simulation_results = pd.concat(
        [
            simulation_results,
            rescale_factor * benchmark_data.rename(columns={c: f'y.benchmark.{c}' for c in benchmark_data.columns})
        ],
        axis=1
    )
    reconstruction_cols = \
        [f'z.baseline.{mtype}{i}' for i, mtype in enumerate(baseline_model_types)] \
        + [f'z.hidden.{mtype}{i}' for i, mtype in enumerate(hidden_model_types)] \
        + ['z.benchmark.' + c for c in benchmark_data.columns]
    reconstruction_data = pd.DataFrame(
        np.concatenate(
            [np.nan * np.ones_like(u_test) for _ in baseline_simulations]
            + hidden_reconstructions
            + [np.nan * np.ones_like(u_test) for _ in benchmark_data.columns],
            axis=1
        ) * rescale_factor,
        columns=reconstruction_cols
    )
    simulation_results = pd.concat([simulation_results, reconstruction_data], axis=1)

    simulation_scores = {
        model_name: {
            'id': f'{model_type_pref}.{model_type_suff}{model_type_idx}',
            'r2': score['r2'],
            'rmse': score['rmse'] * rescale_factor
        } for model_type_pref, (model_type_idx, model_type_suff), model_name, score in zip(
            ['baseline'] * len(baseline_model_types) + ['hidden'] * len(hidden_model_types),
            list(enumerate(baseline_model_types)) + list(enumerate(hidden_model_types)),
            baseline_models_names + hidden_models_names,
            baseline_scores + hidden_scores
        )
    }
    for c in benchmark_data.columns:
        benchmark_scores = dict(
            zip(
                ['r2', 'rmse'], score_simulation(
                    x_test[~benchmark_data[c].isna(), :],
                    benchmark_data.loc[~benchmark_data[c].isna(), c].values
                )
            )
        )
        simulation_scores.update(
            {
                c: {
                    'id': f'benchmark.{c}',
                    'r2': benchmark_scores['r2'],
                    'rmse': benchmark_scores['rmse'] * rescale_factor
                }
            }
        )

    return simulation_results, simulation_scores
