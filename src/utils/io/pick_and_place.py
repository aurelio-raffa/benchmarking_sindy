import numpy as np
import pandas as pd


def prepare_simulation_data(
        t_test,
        x_test,
        u_test,
        model_names,
        simulations,
        scores
):
    simulation_results = pd.DataFrame(
        np.concatenate(
            [t_test.reshape(-1, 1), u_test, x_test] + simulations,
            axis=1
        ),
        columns=['t', 'u', 'y'] + [f'y.{i}' for i, _ in enumerate(model_names)]
    )
    simulation_results = pd.concat([simulation_results], axis=1)

    simulation_scores = {
        model_name: {
            'r2': score['r2'],
            'rmse': score['rmse']
        } for model_name, score in zip(model_names, scores)
    }

    return simulation_results, simulation_scores
