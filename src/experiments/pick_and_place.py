import contextlib
import json
import os
import pickle
import warnings

import hyperopt
import numpy as np
from fire import Fire
from hyperopt import hp, fmin, tpe
from matplotlib import pyplot as plt

from __init__ import root_path
from src.utils.etl import prepare_data
from src.utils.etl.pick_and_place import load_data
from src.utils.io.pick_and_place import prepare_simulation_data
from src.utils.model_selection import train_validate_model
from src.utils.plotting import set_font
from src.utils.plotting.trajectory_plot import trajectory_plot


def pick_and_place(
        output_path: str = 'outputs/pick_and_place',
        trials: int = 1000,
        alpha_lb: float = 1e-6,
        alpha_ub: float = 1e1,
        threshold_lb: float = 1e-6,
        threshold_ub: float = 1e2,
        tvr_gamma_lb: float = 1e-6,
        tvr_gamma_ub: float = 1e1,
        seed: int = 42,
        show_plots: bool = True,
        dt: float = 2.5e-4,
):
    # 00 - Directory setup
    if not os.path.isdir(os.path.join(root_path, output_path)):
        os.makedirs(os.path.join(root_path, output_path), exist_ok=True)

    # 01 - Loading the data
    train_data, validation_data, test_data = load_data()

    # 02 - Training the models
    model_names = ['SINDy', 'SINDy, Second Order']
    model_orders = [1, 2]
    simulations = []
    scores = []

    for readable_name, order in zip(model_names, model_orders):
        print(f' {readable_name} '.center(120, '='))

        def validation_neg_r2(params: dict):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    x_t, u_t, x_dot_t, x_ddot_t = prepare_data(
                        train_data, dt=dt, tvr_gamma=params['tvr_gamma'], derivation_order=2
                    )
                    x_v, u_v, x_dot_v, x_ddot_v = prepare_data(
                        validation_data, dt=dt, tvr_gamma=params['tvr_gamma'], derivation_order=2
                    )
                    t_v = np.arange(len(x_v)) * dt

                    val_neg_r2 = - 1.0 * train_validate_model(
                        t_test=t_v,
                        x_train=x_t,
                        u_train=u_t,
                        x_dot_train=x_dot_t,
                        x_test=x_v,
                        u_test=u_v,
                        x_ddot_train=x_ddot_t,
                        model_type='naive',
                        order=order,
                        alpha=params['alpha'],
                        threshold=params['threshold']
                    )[-2]

            return val_neg_r2

        # Hyperparameter tuning
        rstate = np.random.Generator(np.random.PCG64(seed=seed))
        search_space = {
            'alpha': hp.loguniform('alpha', np.log(alpha_lb), np.log(alpha_ub)),
            'threshold': hp.loguniform('threshold', np.log(threshold_lb), np.log(threshold_ub)),
            'tvr_gamma': hp.loguniform('tvr_gamma', np.log(tvr_gamma_lb), np.log(tvr_gamma_ub))
        }

        best_parameters = hyperopt.space_eval(
            search_space,
            fmin(
                fn=validation_neg_r2,
                space=search_space,
                algo=tpe.suggest,
                max_evals=trials,
                rstate=rstate
            )
        )

        # 02 - Computing the derivatives (using TV regularization) and preparing the dataset
        x_train, u_train, x_dot_train, x_ddot_train = prepare_data(
            train_data, dt=dt, tvr_gamma=best_parameters['tvr_gamma'], derivation_order=2
        )
        x_test, u_test, x_dot_test, x_ddot_test = prepare_data(
            test_data, dt=dt, tvr_gamma=best_parameters['tvr_gamma'], derivation_order=2
        )
        t_test = np.arange(len(x_test)) * dt

        # Fitting
        model, simulation, r2, rmse = train_validate_model(
            t_test=t_test,
            x_train=x_train,
            u_train=u_train,
            x_dot_train=x_dot_train,
            x_test=x_test,
            u_test=u_test,
            x_ddot_train=x_ddot_train,
            model_type='naive',
            order=order,
            alpha=best_parameters['alpha'],
            threshold=best_parameters['threshold']
        )
        model.print()
        print(' Scores: '.center(120, '-'))
        print(f'-> R2: {100 * r2:.2f}%')
        print(f'-> RMSE: {rmse:.2f}')

        simulations.append(simulation[:, -1].reshape(-1, 1))
        scores.append({'r2': r2, 'rmse': rmse})

        if output_path is not None:
            model_savename = f'model_order{order}'
            with open(os.path.join(root_path, output_path, f'{model_savename}_parameters.json'), 'w') as handle:
                json.dump(best_parameters, handle, indent=4)
            with open(os.path.join(root_path, output_path, f'{model_savename}_model.pkl'), 'wb') as handle:
                pickle.dump(model, handle)

    x_test, u_test, _ = prepare_data(test_data, dt=dt)
    t_test = np.arange(len(x_test)) * dt

    # 04 - Pooling all simulation data and saving
    simulation_results, simulation_scores = prepare_simulation_data(
        t_test,
        x_test,
        u_test,
        model_names,
        simulations,
        scores
    )

    if output_path is not None:
        simulation_results.to_csv(os.path.join(root_path, output_path, 'simulation_results.csv'), index=False)
        with open(os.path.join(root_path, output_path, 'simulation_scores.json'), 'w') as handle:
            json.dump(simulation_scores, handle, indent=4)

    # 05 - Plotting
    set_font(publish=bool(os.getenv('PUBLISH')))
    fig = trajectory_plot(
        t=t_test,
        xs=[x_test] + simulations,
        labels=['True'] + model_names,
        x_subplot_label=None,
        x_scale='Displacement',
        scientific_notation=False
    )
    if output_path is not None:
        fig.savefig(
            os.path.join(
                root_path,
                output_path,
                f'test_comparison.pdf'
            ),
            format="pdf",
            bbox_inches="tight"
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    Fire(pick_and_place)
