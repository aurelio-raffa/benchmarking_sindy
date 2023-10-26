import warnings
import hyperopt

import numpy as np
import matplotlib.pyplot as plt

from hyperopt import hp, fmin, tpe
from fire import Fire
from pynumdiff.total_variation_regularization import jerk
from scipy.integrate import solve_ivp

from src.utils.etl import prepare_data
from src.utils.etl.bouc_wen import load_data
from src.utils.model_selection.bouc_wen import train_and_validate_x_model, train_and_validate_z_model
from src.utils.simulation.bouc_wen import ode_model


def bouc_wen(
        output_path: str = None,
        dt: float = 1 / 750.0,
        tvr_gamma: float = 0.000001,
        training_samples: int = 1000,
        validation_samples: int = 500,
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
        'c': hp.loguniform('c', np.log(c_lb), np.log(c_ub)),
        'k': hp.loguniform('k', np.log(k_lb), np.log(k_ub)),
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

    # 04 - Comparison
    # TODO: run model on test data and save metrics!

    _, (ax1, ax3) = plt.subplots(2, 1, sharex=True)

    ax1.plot(t_v, x_v, label='x', zorder=4)
    ax1.plot(t_v, x_z_mod, label='x (simulated)', zorder=3)
    ax1.plot(t_v, x_sindy, label='x (sindy)', zorder=2)
    ax1.legend()

    ax3.plot(t_v, y_v, label='y', zorder=4)
    ax3.plot(t_v, y_z_mod, label='y (simulated)', zorder=3)
    ax3.plot(t_v, y_sindy, label='y (sindy)', zorder=2)
    ax3.legend()

    plt.show()

    # TODO: load validation data
    # TODO: replicate hysteresis loop
    # hysteresis loop plotting, i.e. restoring force vs. displacement
    # xd_ = (u_ - k * y_ - c * x_ - z_) / m

    n_qs = 5000
    t_qs = np.arange(n_qs) * dt
    # u_qs = 150.0 * np.concatenate(
    #     [
    #         1 - np.exp(- 1e-2 * t_qs[:n_qs // 2]),
    #         2 * (np.exp(- 1e-2 * (t_qs[n_qs // 2:] - t_qs[n_qs // 2])) - 0.5)
    #     ]
    # ).reshape(-1, 1)
    u_qs = 150.0 * np.sin(np.linspace(0.0, 2 * np.pi * dt * n_qs, n_qs)).reshape(-1, 1)
    # u_qs = 10.0 * np.concatenate(
    #     [
    #         np.sin(np.linspace(0.0, 0.5 * np.pi, n_qs // 5)),
    #         np.ones((n_qs // 5,)),
    #         np.sin(np.linspace(0.5 * np.pi, 1.5 * np.pi, n_qs // 5)),
    #         - 1.0 * np.ones((n_qs // 5,)),
    #         np.sin(np.linspace(1.5 * np.pi, 2 * np.pi, n_qs // 5))
    #     ]
    # ).reshape(-1, 1)
    qs_model = ode_model(u_qs, t_qs, x_grad=m2, z_grad=m1)

    # simulating
    xyz_qs = solve_ivp(qs_model, [0.0, t_qs[-1]], [0.0, 0.0, 0.0], t_eval=t_qs)['y']
    x_qs = xyz_qs[0, :].reshape(-1, 1)
    y_qs = xyz_qs[1, :].reshape(-1, 1)

    f = - 1.0 * jerk(x_qs.ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1) * m2_coef[0, -1] + u_qs

    xy_sindy_qs = m0.simulate(np.array([0.0, 0.0]), t_qs, u=u_qs)
    x_sindy_qs = np.pad(xy_sindy_qs[:, 0], (0, 1), mode='edge').reshape(-1, 1)
    y_sindy_qs = np.pad(xy_sindy_qs[:, 1], (0, 1), mode='edge').reshape(-1, 1)

    m0_coef = m0.coefficients().copy()
    f_sindy = - 1.0 * jerk(x_sindy_qs.ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1) * m2_coef[0, 2] + u_qs

    plt.figure()
    # plt.plot(y_qs, f, label='z model')
    # plt.plot(y_sindy_qs, f_sindy, label='sindy model')
    plt.plot(u_qs, y_qs, label='z model')
    plt.plot(u_qs, y_sindy_qs, label='sindy model')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    Fire(bouc_wen)
