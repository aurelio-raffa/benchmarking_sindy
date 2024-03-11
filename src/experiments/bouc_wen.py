import contextlib
import json
import pickle
import warnings
import os
from copy import deepcopy

import hyperopt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hyperopt import hp, fmin, tpe
from fire import Fire
from scipy.integrate import solve_ivp

from __init__ import root_path
from src.utils.plotting import set_font
from src.utils.preprocessing.compute_derivatives import compute_derivatives

from src.utils.etl import prepare_data
from src.utils.etl.bouc_wen import load_data
from src.utils.model_selection.bouc_wen import train_and_validate_x_model, train_and_validate_z_model
from src.utils.plotting.hysteresis_plot import hysteresis_plot
from src.utils.plotting.trajectory_plot import trajectory_plot
from src.utils.simulation.bouc_wen import simulate_test, high_fidelity


def bouc_wen(
        output_path: str = 'outputs/bouc_wen',
        dt: float = 1 / 750.0,
        tvr_gamma: float = 1e-8,
        training_samples: int = 40000,
        validation_samples: int = 960,
        low_frequency_samples: int = 1000,
        naive_model_selection_trials: int = 25,
        hidden_model_ideal_selection_trials: int = 50,
        hidden_model_full_selection_trials: int = 100,
        m_mean: float = 2.25,
        m_std: float = 0.77,
        c_mean: float = 9.75,
        c_std: float = 0.77,
        k_mean: float = 5.0123e4,
        k_std: float = 0.77,
        x_threshold_lb: float = 1e-4,
        x_threshold_ub: float = 1.0,
        z_threshold_lb: float = 1e-2,
        z_threshold_ub: float = 1e2,
        tvr_gamma_lb: float = 1e-8,
        tvr_gamma_ub: float = 1e-0,
        alpha_lb: float = 1e-3,
        alpha_ub: float = 1e2,
        verbose: bool = True,
        show_plots: bool = False,
        seed: int = 42,
):
    """Compares a SINDy naive model with a more sophisticated approach on the Bouc-Wen hysteresis benchmark.
    """
    # 00 - Directory setup
    if not os.path.isdir(os.path.join(root_path, output_path)):
        os.makedirs(os.path.join(root_path, output_path), exist_ok=True)

    # 01 - Loading the data
    training_data, valid_data, test1_data, test2_data = load_data(
        training_samples=training_samples,
        validation_samples=validation_samples,
    )

    # 02 - Naive SINDy model
    def x_model_validation_rmse(params: dict):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(open(os.devnull, 'w')):

                val_rmse = train_and_validate_x_model(
                    training_data,
                    valid_data,
                    verbose=False,
                    **params
                )[3]

        return val_rmse

    rstate = np.random.Generator(np.random.PCG64(seed=seed))
    x_model_search_space = {
        'threshold': hp.loguniform('x_threshold', np.log(x_threshold_lb), np.log(x_threshold_ub)),
        'alpha': hp.loguniform('alpha', np.log(alpha_lb), np.log(alpha_ub)),
        'tvr_gamma': hp.loguniform('tvr_gamma', np.log(tvr_gamma_lb), np.log(tvr_gamma_ub)),
    }

    if verbose:
        print(' Tuning SINDy naive model '.center(120, '='))
    x_model_best_parameters = hyperopt.space_eval(
        x_model_search_space,
        fmin(
            fn=x_model_validation_rmse,
            space=x_model_search_space,
            algo=tpe.suggest,
            max_evals=naive_model_selection_trials,
            rstate=rstate
        )
    )

    m0, x_sindy, y_sindy, sindy_rmse, sindy_r2 = train_and_validate_x_model(
        training_data,
        valid_data,
        verbose=verbose,
        **x_model_best_parameters
    )

    if output_path is not None:
        with open(os.path.join(root_path, output_path, 'naive_model_parameters.json'), 'w') as handle:
            json.dump(x_model_best_parameters, handle, indent=4)
        with open(os.path.join(root_path, output_path, 'naive_model_m0.pkl'), 'wb') as handle:
            pickle.dump(m0, handle)

    # 03 - Fitting the SINDy model(s) for the hidden state
    def z_model_validation_rmse(params: dict):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    val_rmse = train_and_validate_z_model(
                        training_data,
                        valid_data,
                        dt=dt,
                        verbose=False,
                        **params
                    )[-2]
        except Exception as e:
            # if anything fails in the training, we return the worst possible value
            warnings.warn(f'Unexpected error in trial ({type(e).__name__}: {e}), returning Inf...')
            val_rmse = np.inf

        return val_rmse

    # Notice that it is also possible to tune parameter nu, which we do not do here
    rstate = np.random.Generator(np.random.PCG64(seed=seed))
    z_model_ideal_search_space = {
        'tvr_gamma': hp.loguniform('tvr_gamma', np.log(tvr_gamma_lb), np.log(tvr_gamma_ub)),
        'tvr_gamma_z': hp.loguniform('tvr_gamma_z', np.log(tvr_gamma_lb), np.log(tvr_gamma_ub)),
        'z_threshold': hp.loguniform('z_threshold', np.log(z_threshold_lb), np.log(z_threshold_ub))
    }
    z_model_full_search_space = {
        'm': hp.lognormal('m', np.log(m_mean), m_std),
        'c': hp.lognormal('c', np.log(c_mean), c_std),
        'k': hp.lognormal('k', np.log(k_mean), k_std),
    }
    z_model_full_search_space.update(deepcopy(z_model_ideal_search_space))

    hidden_models = {}
    for name, [search_space, trials] in {
        'ideal': [z_model_ideal_search_space, hidden_model_ideal_selection_trials],
        'HPO': [z_model_full_search_space, hidden_model_full_selection_trials]
    }.items():
        if verbose:
            print(f' Tuning SINDy full model ({name} parameters) '.center(120, '='))

        best_parameters = hyperopt.space_eval(
            search_space,
            fmin(
                fn=z_model_validation_rmse,
                space=search_space,
                algo=tpe.suggest,
                max_evals=trials,
                rstate=rstate
            )
        )

        m1, m2_coef, x_z_mod, y_z_mod, z_z_mod, z_mod_rmse, z_mod_r2 = train_and_validate_z_model(
            training_data,
            valid_data,
            dt=dt,
            verbose=verbose,
            **best_parameters
        )
        hidden_models[name] = {
            'm1': m1,
            'm2_coef': m2_coef,
            'x_z_mod': x_z_mod,
            'y_z_mod': y_z_mod,
            'z_z_mod': z_z_mod,
            'z_mod_rmse': z_mod_rmse,
            'z_mod_r2': z_mod_r2
        }

        if output_path is not None:
            with open(os.path.join(root_path, output_path, f'hidden_model_{name}_parameters.json'), 'w') as handle:
                json.dump(best_parameters, handle, indent=4)
            with open(os.path.join(root_path, output_path, f'hidden_model_{name}_m1.pkl'), 'wb') as handle:
                pickle.dump(m1, handle)

    # 04 - Preparing test data
    #   - Low-frequency simulation
    #   - Multisine excitation
    #   - Sine sweep excitation
    n_low_freq = low_frequency_samples
    t_low_freq = np.arange(n_low_freq) * dt
    u_low_freq = 150.0 * np.sin(np.linspace(0.0, 2 * np.pi * dt * n_low_freq, n_low_freq)).reshape(-1, 1)
    hf_model = high_fidelity(t_low_freq, u_low_freq)

    xyz_hf = solve_ivp(hf_model, [0.0, t_low_freq[-1]], [0.0, 0.0, 0.0], t_eval=t_low_freq, method='LSODA')['y']
    x_low_freq = xyz_hf[0, :].reshape(-1, 1)
    y_low_freq = xyz_hf[1, :].reshape(-1, 1)
    xd_low_freq = compute_derivatives(x_low_freq.ravel(), dt=dt, tvr_gamma=tvr_gamma, order=1)[0].reshape(-1, 1)

    # 05 - Comparison on test data
    for i, ((y_test, u_test, x_test, xd_test), test_name) in enumerate(
            zip(
                [
                    prepare_data(valid_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2),
                    (y_low_freq, u_low_freq, x_low_freq, xd_low_freq),
                    prepare_data(test1_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2),
                    prepare_data(test2_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2)
                ],
                [
                    'validation',
                    'lowfrequency',
                    'multisine',
                    'sinesweep'
                ]
            )
    ):
        if verbose:
            print(f' Testing on set "{test_name}" '.center(120, '='))

        # running the simulation
        t_test, x_sim_test, y_sim_test, r2_test, rmse_test = simulate_test(
            y_test, u_test, x_test, xd_test,
            [m0],
            [(model['m1'], model['m2_coef']) for model in hidden_models.values()],
            dt=dt,
            verbose=verbose
        )

        # saving the results
        if output_path is not None:
            test_outputs = pd.DataFrame(
                np.concatenate(
                    [u_test, y_test, *y_sim_test, x_test, *x_sim_test],
                    axis=1
                ),
                columns=[
                    'u',
                    'y', 'y_naive', *[f'y_hidden_{n}' for n in hidden_models.keys()],
                    'x', 'x_naive', *[f'x_hidden_{n}' for n in hidden_models.keys()]
                ]
            )
            test_outputs.to_csv(
                os.path.join(
                    root_path,
                    output_path,
                    f'test_{test_name}_outputs.csv'
                ),
                index=False
            )
            with open(os.path.join(root_path, output_path, f'test_{test_name}_scores.json'), 'w') as handle:
                json.dump(
                    {
                        'naive': {
                            'r2': float(r2_test[0]),
                            'rmse': float(rmse_test[0])
                        },
                        **{
                            f'hidden_{n}': {
                                'r2': float(r2_test[j + 1]),
                                'rmse': float(rmse_test[j + 1])
                            } for j, n in enumerate(hidden_models.keys())
                        }
                    },
                    handle
                )

        if verbose:
            print(f'SINDy naive model:')
            print(f'\tR2: {100 * r2_test[0]:.3f}%, RMSE: {rmse_test[0]:.3e}')
            for j, n in enumerate(hidden_models.keys()):
                print(f'SINDy hidden {n} model:')
                print(f'\tR2: {100 * r2_test[j + 1]:.3f}%, RMSE: {rmse_test[j + 1]:.3e}')

        # plotting the trajectories
        plot_labels = ['True', 'SINDy', *[f'Hidden SINDy, {n}' for n in hidden_models.keys()]]
        set_font(publish=bool(os.getenv('PUBLISH')))
        fig = trajectory_plot(
            t=t_test,
            xs=[y_test, *y_sim_test],
            labels=plot_labels,
            x_subplot_label=None,
            x_scale='Displacement [m]'
        )

        if output_path is not None:
            fig.savefig(
                os.path.join(
                    root_path,
                    output_path,
                    f'test_{test_name}_trajectories.pdf'
                ),
                format="pdf",
                bbox_inches="tight"
            )
        if show_plots:
            plt.show()
        plt.close(fig)

        # plotting the displacement vs input force
        fig = hysteresis_plot(
            u_test,
            [y_test, *y_sim_test],
            labels=plot_labels
        )

        if output_path is not None:
            fig.savefig(
                os.path.join(
                    root_path,
                    output_path,
                    f'test_{test_name}_hysteresis.pdf'
                ),
                format="pdf",
                bbox_inches="tight"
            )
        if show_plots:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    Fire(bouc_wen)
