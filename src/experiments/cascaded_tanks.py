"""Code based on https://pysindy.readthedocs.io/en/latest/examples/9_sindypi_with_sympy/example.html#Find-complex-PDE-with-SINDy-PI-with-PDE-functionality
"""
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
from src.utils.etl.cascaded_tanks import load_data
from src.utils.io.cascaded_tanks import prepare_simulation_data
from src.utils.model_selection.cascaded_tanks import train_validate_hidden_model
from src.utils.model_selection import train_validate_model
from src.utils.benchmarking.cascaded_tanks import load_benchmarks
from src.utils.plotting.cascaded_tanks.plot_top_models import plot_top_models


def cascaded_tanks(
        output_path: str = 'outputs/cascaded_tanks',
        rescale_factor: float = 10.0,
        tvr_gamma: float = 1.0,
        state_space_models_trials: int = 100,
        hidden_models_trials: int = 1000,
        k1_mean: float = 16.0,
        k1_log_std: float = 0.5,
        k3_mean: float = 45.0,
        k3_log_std: float = 0.5,
        z0_lb: float = 0.01,
        z0_ub: float = 0.99,
        alpha_lb: float = 1e-3,
        alpha_ub: float = 1e3,
        threshold_lb: float = 1e-6,
        threshold_ub: float = 1e2,
        seed: int = 42,
        show_plots: bool = True,
        top_n: int = 4
):
    # 00 - Directory setup
    if not os.path.isdir(os.path.join(root_path, output_path)):
        os.makedirs(os.path.join(root_path, output_path), exist_ok=True)

    # 01 - Loading the data
    train_data, validation_data, test_data, dt, t_train, t_val, t_test = load_data()
    benchmark_data = load_benchmarks()

    # 01b - Rescaling the data (for numerical stability)
    train_data /= rescale_factor
    validation_data /= rescale_factor
    test_data /= rescale_factor
    benchmark_data /= rescale_factor

    # 02 - Computing the derivatives (using TV regularization) and preparing the dataset
    x_train, u_train, x_dot_train, x_ddot_train = prepare_data(
        train_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2
    )
    x_val, u_val, x_dot_val, x_ddot_val = prepare_data(
        validation_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2
    )
    x_test, u_test, x_dot_test, x_ddot_test = prepare_data(
        test_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2
    )

    # 03a - Training the baseline models
    baseline_models_names = ['SINDy, Poly.', 'SINDy, Sqrt.', 'SINDy 2nd order, Poly.', 'SINDy 2nd order, Sqrt.']
    baseline_model_types = ['naive', 'sqrt_poly', 'naive', 'sqrt_poly']
    baseline_simulations = []
    baseline_scores = []

    for readable_name, model_type, order, integrator_kws in zip(
            baseline_models_names,
            baseline_model_types,
            [1, 1, 2, 2],
            [{}, {}, {'method': 'Radau'}, {'method': 'Radau'}]
    ):
        print(f' {readable_name} '.center(120, '='))

        def validation_rmse(params: dict):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with contextlib.redirect_stderr(open(os.devnull, 'w')):

                    val_rmse = train_validate_model(
                        t_test=t_val,
                        x_train=x_train,
                        u_train=u_train,
                        x_dot_train=x_dot_train,
                        x_test=x_val,
                        u_test=u_val,
                        x_ddot_train=x_ddot_train,
                        model_type=model_type,
                        order=order,
                        integrator_kws=integrator_kws,
                        **params
                    )[-1]

            return val_rmse

        # Hyperparameter tuning
        rstate = np.random.Generator(np.random.PCG64(seed=seed))
        search_space = {
            'alpha': hp.loguniform('alpha', np.log(alpha_lb), np.log(alpha_ub)),
            'threshold': hp.loguniform('threshold', np.log(threshold_lb), np.log(threshold_ub))
        }

        best_parameters = hyperopt.space_eval(
            search_space,
            fmin(
                fn=validation_rmse,
                space=search_space,
                algo=tpe.suggest,
                max_evals=state_space_models_trials,
                rstate=rstate
            )
        )

        # Fitting
        model, simulation, r2, rmse = train_validate_model(
            t_test=t_test,
            x_train=x_train,
            u_train=u_train,
            x_dot_train=x_dot_train,
            x_test=x_test,
            u_test=u_test,
            x_ddot_train=x_ddot_train,
            model_type=model_type,
            order=order,
            integrator_kws=integrator_kws,
            **best_parameters
        )
        model.print()
        print(' Scores: '.center(120, '-'))
        print(f'-> R2: {100 * r2:.2f}%')
        print(f'-> RMSE: {rescale_factor * rmse:.2f}')

        baseline_simulations.append(simulation[:, -1].reshape(-1, 1))
        baseline_scores.append({'r2': r2, 'rmse': rmse})

        if output_path is not None:
            model_savename = f'baseline_{model_type}_order{order}'
            with open(os.path.join(root_path, output_path, f'{model_savename}_parameters.json'), 'w') as handle:
                json.dump(best_parameters, handle, indent=4)
            with open(os.path.join(root_path, output_path, f'{model_savename}_model.pkl'), 'wb') as handle:
                pickle.dump(model, handle)

    # 03b - Training the models based on hidden state reconstruction
    hidden_models_names = ['Hidden SINDy, Sqrt.', 'Hidden SINDy, Poly.']
    hidden_model_types = ['sqrt', 'naive']
    hidden_simulations = []
    hidden_reconstructions = []
    hidden_scores = []

    for readable_name, model_type, integrator_kws in zip(
            hidden_models_names,
            hidden_model_types,
            [{'method': 'Radau'}, {'method': 'Radau'}]
    ):
        print(f' {readable_name} '.center(120, '='))

        def validation_rmse(params: dict):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    val_rmse = train_validate_hidden_model(
                        t_train=t_train,
                        t_test=t_val,
                        x_train=x_train,
                        u_train=u_train,
                        x_dot_train=x_dot_train,
                        x_test=x_val,
                        u_test=u_val,
                        x_dot_test=x_dot_val,
                        model_type=model_type,
                        integrator_kws=integrator_kws,
                        **params
                    )[-1]

            return val_rmse

        # Hyperparameter tuning
        rstate = np.random.Generator(np.random.PCG64(seed=seed))
        search_space = {
            'alpha': hp.loguniform('alpha', np.log(alpha_lb), np.log(alpha_ub)),
            'threshold': hp.loguniform('threshold', np.log(threshold_lb), np.log(threshold_ub)),
            'k1': hp.lognormal('k1', np.log(k1_mean), k1_log_std),
            'k3': hp.lognormal('k3', np.log(k3_mean), k3_log_std),
            'z0': hp.uniform('z0', z0_lb, z0_ub),
        }

        best_parameters = hyperopt.space_eval(
            search_space,
            fmin(
                fn=validation_rmse,
                space=search_space,
                algo=tpe.suggest,
                max_evals=hidden_models_trials,
                rstate=rstate
            )
        )

        # Fitting
        model, simulation, hidden, r2, rmse = train_validate_hidden_model(
            t_train=t_train,
            t_test=t_test,
            x_train=x_train,
            u_train=u_train,
            x_dot_train=x_dot_train,
            x_test=x_test,
            u_test=u_test,
            x_dot_test=x_dot_test,
            model_type=model_type,
            integrator_kws=integrator_kws,
            **best_parameters
        )
        model.print()
        print(' Scores: '.center(120, '-'))
        print(f'-> R2: {100 * r2:.2f}%')
        print(f'-> RMSE: {rescale_factor * rmse:.2f}')

        hidden_simulations.append(simulation)
        hidden_reconstructions.append(hidden)
        hidden_scores.append({'r2': r2, 'rmse': rmse})

        if output_path is not None:
            model_savename = f'hidden_{model_type}'
            with open(os.path.join(root_path, output_path, f'{model_savename}_parameters.json'), 'w') as handle:
                json.dump(best_parameters, handle, indent=4)
            with open(os.path.join(root_path, output_path, f'{model_savename}_model.pkl'), 'wb') as handle:
                pickle.dump(model, handle)

    # 04 - Pooling all simulation data and saving
    simulation_results, simulation_scores = prepare_simulation_data(
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
        rescale_factor=rescale_factor
    )

    if output_path is not None:
        simulation_results.to_csv(os.path.join(root_path, output_path, 'simulation_results.csv'), index=False)
        with open(os.path.join(root_path, output_path, 'simulation_scores.json'), 'w') as handle:
            json.dump(simulation_scores, handle, indent=4)

    # 05 - Plotting
    fig = plot_top_models(
        simulation_results,
        simulation_scores,
        top_n=top_n
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
    Fire(cascaded_tanks)
