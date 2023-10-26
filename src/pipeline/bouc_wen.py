import json
import pickle
import warnings
import os
import hyperopt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hyperopt import hp, fmin, tpe
from fire import Fire
from scipy.integrate import solve_ivp

from __init__ import root_path
from src.utils.preprocessing.compute_derivatives import compute_derivatives

from src.utils.etl import prepare_data
from src.utils.etl.bouc_wen import load_data
from src.utils.model_selection.bouc_wen import train_and_validate_x_model, train_and_validate_z_model
from src.utils.plotting.hysteresis_plot import hysteresis_plot
from src.utils.plotting.trajectory_plot import trajectory_plot
from src.utils.simulation.bouc_wen import simulate_test, high_fidelity


# TODO: save all outputs
# TODO: prepare directories for saving outputs
def bouc_wen(
        output_path: str = 'outputs',
        dt: float = 1 / 750.0,
        tvr_gamma: float = 0.000001,
        training_samples: int = 1000,
        validation_samples: int = 500,
        low_frequency_samples: int = 1000,
        model_selection_trials: int = 10,
        alternating_iterations: int = 1,
        m_lb: float = 0.2,
        m_ub: float = 20.0,
        c_lb: float = 1.0,
        c_ub: float = 100.0,
        k_lb: float = 1e4,
        k_ub: float = 1e6,
        x_threshold_lb: float = 1e-4,
        x_threshold_ub: float = 1.0,
        z_threshold_lb: float = 1e-2,
        z_threshold_ub: float = 1e2,
        alpha_lb: float = 1e-3,
        alpha_ub: float = 1e2,
        verbose: bool = True,
        show_plots: bool = False,
        seed: int = 42,
):
    """Compares a SINDy naive model with a more sophisticated approach on the Bouc-Wen hysteresis benchmark.
    """
    # 01 - Loading the data
    training_data, validation_data, test1_data, test2_data = load_data(
        training_samples=training_samples,
        validation_samples=validation_samples,
    )
    y_t, u_t, x_t, xd_t = prepare_data(training_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2)
    y_v, u_v, x_v, xd_v = prepare_data(validation_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2)
    t_t = np.arange(0, y_t.shape[0]) * dt
    t_v = np.arange(0, y_v.shape[0]) * dt

    # 02 - Naive SINDy model
    def x_model_validation_rmse(params: dict):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_rmse = train_and_validate_x_model(
                t_t, y_t, x_t, u_t, xd_t,
                t_v, y_v, x_v, u_v, xd_v,
                verbose=False,
                **params
            )[3]

        return val_rmse

    rstate = np.random.Generator(np.random.PCG64(seed=seed))
    x_model_search_space = {
        'threshold': hp.loguniform('x_threshold', np.log(x_threshold_lb), np.log(x_threshold_ub)),
        'alpha': hp.loguniform('alpha', np.log(alpha_lb), np.log(alpha_ub)),
    }

    if verbose:
        print(' Tuning SINDy naive model '.center(120, '='))
    x_model_best_parameters = hyperopt.space_eval(
        x_model_search_space,
        fmin(
            fn=x_model_validation_rmse,
            space=x_model_search_space,
            algo=tpe.suggest,
            max_evals=model_selection_trials,
            rstate=rstate
        )
    )

    m0, x_sindy, y_sindy, sindy_rmse, sindy_r2 = train_and_validate_x_model(
        t_t, y_t, x_t, u_t, xd_t,
        t_v, y_v, x_v, u_v, xd_v,
        verbose=verbose,
        **x_model_best_parameters
    )

    if output_path is not None:
        with open(os.path.join(root_path, output_path, 'naive_model_m0.pkl'), 'wb') as handle:
            pickle.dump(m0, handle)

    # 03 - Fitting the SINDy model for the hidden state
    def z_model_validation_rmse(params: dict):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_rmse = train_and_validate_z_model(
                t_t, y_t, x_t, u_t, xd_t,
                t_v, y_v, x_v, u_v, xd_v,
                alternating_iterations=alternating_iterations,
                tvr_gamma=tvr_gamma,
                dt=dt,
                verbose=False,
                **params
            )[6]

        return val_rmse

    # Parameter name    | Lower bound   | Upper bound
    # ------------------|---------------|-------------
    # mL                | 0.2 kg        | 20 kg
    # cL                | 1 N/(m/s)     | 100 N/(m/s)
    # kL                | 1 x 10^4 N/m  | 1 x 10^6 N/m
    # ------------------------------------------------
    # from https://doi.org/10.1016/j.ymssp.2018.04.001
    # Notice that it is also possible to tune parameter nu, which we do not do here
    rstate = np.random.Generator(np.random.PCG64(seed=seed))
    z_model_search_space = {
        'm': hp.loguniform('m', np.log(m_lb), np.log(m_ub)),
        'c': hp.loguniform('c', np.log(k_lb), np.log(k_ub)),
        'k': hp.loguniform('k', np.log(c_lb), np.log(c_ub)),
        'z_threshold': hp.loguniform('z_threshold', np.log(z_threshold_lb), np.log(z_threshold_ub))
    }

    if verbose:
        print(' Tuning SINDy hidden model '.center(120, '='))
    z_model_best_parameters = hyperopt.space_eval(
        z_model_search_space,
        fmin(
            fn=z_model_validation_rmse,
            space=z_model_search_space,
            algo=tpe.suggest,
            max_evals=model_selection_trials,
            rstate=rstate
        )
    )

    m1, m2, m2_coef, x_z_mod, y_z_mod, z_z_mod, z_mod_rmse, z_mod_r2, z, z_est = train_and_validate_z_model(
        t_t, y_t, x_t, u_t, xd_t,
        t_v, y_v, x_v, u_v, xd_v,
        alternating_iterations=alternating_iterations,
        tvr_gamma=tvr_gamma,
        dt=dt,
        verbose=verbose,
        **z_model_best_parameters
    )

    if output_path is not None:
        with open(os.path.join(root_path, output_path, 'hidden_model_m1.pkl'), 'wb') as handle:
            pickle.dump(m1, handle)
        with open(os.path.join(root_path, output_path, 'hidden_model_m2.pkl'), 'wb') as handle:
            pickle.dump(m2, handle)

    # 04 - Preparing test data
    #   - Low-frequency simulation
    #   - Multisine excitation
    #   - Sine sweep excitation
    if verbose:
        print(f' Low-frequency excitation '.center(120, '='))
    n_low_freq = low_frequency_samples
    t_low_freq = np.arange(n_low_freq) * dt
    u_low_freq = 150.0 * np.sin(np.linspace(0.0, 2 * np.pi * dt * n_low_freq, n_low_freq)).reshape(-1, 1)
    hf_model = high_fidelity(t_low_freq, u_low_freq)

    xyz_hf = solve_ivp(hf_model, [0.0, t_low_freq[-1]], [0.0, 0.0, 0.0], t_eval=t_low_freq, method='LSODA')['y']
    x_low_freq = xyz_hf[0, :].reshape(-1, 1)
    y_low_freq = xyz_hf[1, :].reshape(-1, 1)
    xd_low_freq = compute_derivatives(x_low_freq.ravel(), dt=dt, tvr_gamma=tvr_gamma, order=1)[0].reshape(-1, 1)

    y_test1, u_test1, x_test1, xd_test1 = prepare_data(test1_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2)
    y_test2, u_test2, x_test2, xd_test2 = prepare_data(test2_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2)

    # 05 - Comparison on test data
    for i, ((y_test, u_test, x_test, xd_test), test_name) in enumerate(
            zip(
                [
                    (y_low_freq, u_low_freq, x_low_freq, xd_low_freq),
                    (y_test1, u_test1, x_test1, xd_test1),
                    (y_test2, u_test2, x_test2, xd_test2)
                ],
                [
                    'lowfrequency',
                    'multisine',
                    'sinesweep'
                ]
            )
    ):
        if verbose:
            print(f' Test set "{test_name}" '.center(120, '='))

        # running the simulation
        (
            t_test, x_sindy_test, y_sindy_test, x_hidden_test, y_hidden_test,
            sindy_r2_test, hidden_r2_test, sindy_rmse_test, hidden_rmse_test
        ) = simulate_test(
            y_test, u_test, x_test, xd_test,
            m0, m1, m2,
            dt=dt
        )

        # saving the results
        if output_path is not None:
            test_outputs = pd.DataFrame(
                np.concatenate(
                    [x_test, y_test, u_test, x_sindy_test, y_sindy_test, x_hidden_test, y_hidden_test],
                    axis=1
                ),
                columns=['x', 'y', 'u', 'x_naive', 'y_naive', 'x_hidden', 'y_hidden']
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
                            'r2': float(sindy_r2_test),
                            'rmse': float(sindy_rmse_test)
                        },
                        'hidden': {
                            'r2': float(hidden_r2_test),
                            'rmse': float(hidden_rmse_test)
                        }
                    },
                    handle
                )

        if verbose:
            print(f'SINDy naive model:')
            print(f'\tR2: {100 * sindy_r2_test:.3f}%, RMSE: {sindy_rmse_test:.3e}')
            print(f'SINDy hidden model:')
            print(f'\tR2: {100 * hidden_r2_test:.3f}%, RMSE: {hidden_rmse_test:.3e}')

        # plotting the trajectories
        fig = trajectory_plot(
            t_test,
            u_test,
            [x_test, x_hidden_test, x_sindy_test],
            [y_test, y_hidden_test, y_sindy_test],
            labels=['True', 'SINDy hidden', 'SINDy naive']
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
            [y_test, y_hidden_test, y_sindy_test],
            labels=['True', 'SINDy hidden', 'SINDy naive']
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


if __name__ == '__main__':
    Fire(bouc_wen)
