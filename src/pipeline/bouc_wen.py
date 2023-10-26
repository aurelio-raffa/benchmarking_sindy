import os
import logging
import warnings
from copy import deepcopy
import hyperopt

import pysindy as ps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hyperopt import hp, fmin, tpe
from typing import Tuple, Any
from fire import Fire
from pynumdiff.total_variation_regularization import jerk
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score


from __init__ import root_path
from src.utils.etl import prepare_data
from src.utils.etl.bouc_wen import load_data


silence_loggers = [
    "hyperopt.tpe",
    "hyperopt.fmin",
    "hyperopt.pyll.base",
]
for logger in silence_loggers:
    logging.getLogger(logger).setLevel(logging.ERROR)


def ode_model(u: np.ndarray, t: np.ndarray, x_grad: Any, z_grad: Any):
    # preparing the model for simulation
    def mod(t_, xyz_):
        [x_, y_, z_] = xyz_
        u_ = np.interp(t_, t, u.ravel())

        zd_ = z_grad.predict(np.array([[z_]]), u=np.array([[x_]])).ravel()[0]
        xd_ = x_grad.predict(np.array([[x_]]), u=np.array([[y_, z_, u_]])).ravel()[0]
        yd_ = x_

        return np.array([xd_, yd_, zd_])

    return mod


def xdot_model(include_z: bool = True, degree: int = 3, threshold: float = 0.00001, alpha: float = 10.0):
    """simple library consisting of a linear combination of inputs and state
    """
    if include_z:
        feature_names = ['x', 'y', 'z', 'u']
    else:
        feature_names = ['x', 'y', 'u']

    lib = ps.PolynomialLibrary(
        include_bias=False,
        degree=degree,
    )
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        feature_names=feature_names,
        differentiation_method=None,
    )
    return model, feature_names


def zdot_model(nu: float = 1.0, z_term: bool = True, threshold: float = 1.0, alpha: float = 10.0):
    """library consisting of transformations akin the Bouc-Wen model
    """
    feature_names = ['z', 'x']
    if nu != 1.0:
        functions = [
            (lambda x: x) if z_term else (lambda _, x: x),
            lambda z, x: np.power(np.abs(z), nu) * np.abs(x) * np.sign(z),
            lambda z, x: np.power(np.abs(z), nu) * x,
            lambda z, x: np.power(np.abs(z), nu) * np.abs(x)
        ]
        function_names = [
            (lambda x: x) if z_term else (lambda _, x: x),
            lambda z, x: f'|{z}|^{nu:.2f} * |{x}| * sign({z})',
            lambda z, x: f'|{z}|^{nu:.2f} * {x}',
            lambda z, x: f'|{z}|^{nu:.2f} * |{x}|'
        ]
    else:
        functions = [
            (lambda x: x) if z_term else (lambda _, x: x),
            lambda z, x: z * np.abs(x),
            lambda z, x: np.abs(z) * x,
            lambda z, x: np.abs(z) * np.abs(x)
        ]
        function_names = [
            (lambda x: x) if z_term else (lambda _, x: x),
            lambda z, x: f'{z} * |{x}|',
            lambda z, x: f'|{z}| * {x}',
            lambda z, x: f'|{z}| * |{x}|'
        ]
    lib = ps.CustomLibrary(
        library_functions=functions,
        function_names=function_names,
        include_bias=False
    )
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        feature_names=feature_names,
        differentiation_method=None,
    )
    return model, feature_names


def fit_x_model(
        t_t: np.ndarray,
        y_t: np.ndarray,
        x_t: np.ndarray,
        u_t: np.ndarray,
        xd_t: np.ndarray,
        t_v: np.ndarray,
        y_v: np.ndarray,
        x_v: np.ndarray,
        u_v: np.ndarray,
        xd_v: np.ndarray,
        verbose: bool = True,
        threshold: float = 0.00001,
        alpha: float = 10.0,
):
    m0, _ = xdot_model(include_z=False, threshold=threshold, alpha=alpha)
    m0.fit(np.concatenate([x_t, y_t], axis=1), u=u_t, x_dot=np.concatenate([xd_t, x_t], axis=1))

    try:
        xy_sindy = m0.simulate(np.array([x_v[0, 0], y_v[0, 0]]), t_v, u=u_v)
        x_sindy = np.pad(xy_sindy[:, 0], (0, 1), mode='edge').reshape(-1, 1)
        y_sindy = np.pad(xy_sindy[:, 1], (0, 1), mode='edge').reshape(-1, 1)
        sindy_r2 = r2_score(y_v.ravel(), y_sindy.ravel())
        sindy_rmse = np.sqrt(np.mean((y_v.ravel() - y_sindy.ravel()) ** 2))

        if verbose:
            print('=' * 120)
            print(f'SINDy naive model:')
            m0.print()
            print('-' * 120)
            print(f'\tR2: {100 * sindy_r2:.3f}%, RMSE: {sindy_rmse:.3e}')

        return m0, x_sindy, y_sindy, sindy_rmse, sindy_r2
    except Exception:
        warnings.warn('SINDy naive model failed, terminating prematurely...')
        return None, None, None, np.inf, -np.inf


def fit_z_model(
        t_t: np.ndarray,
        y_t: np.ndarray,
        x_t: np.ndarray,
        u_t: np.ndarray,
        xd_t: np.ndarray,
        t_v: np.ndarray,
        y_v: np.ndarray,
        x_v: np.ndarray,
        u_v: np.ndarray,
        xd_v: np.ndarray,
        k: float = 5e4,
        c: float = 10.0,
        m: float = 2.0,
        nu: float = 1.0,
        alternating_iterations: int = 1,
        tvr_gamma: float = 0.000001,
        dt: float = 1 / 750.0,
        verbose: bool = True,
        x_threshold: float = 1e-6,
        z_threshold: float = 1.0,
        z_term: bool = True,
):
    # We use the first model equation to recover the hidden state z from the initial guess
    # xd = (u - k * y - c * x - z) / m
    z_est = u_t - k * y_t - c * x_t - m * xd_t
    z_ = z_est.copy()
    m1, m2, m2_coef = None, None, None
    x_full, y_full, z_full = None, None, None
    z_mod_rmse, z_mod_r2, z = np.inf, - np.inf, None
    for it in range(alternating_iterations):
        if verbose:
            print(f' Iteration {it + 1} '.center(120, '='))

        # First stage: we use the z estimate to fit a dynamical model based on perfect knowledge of x_dot
        zd = jerk(z_.ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1)
        m1_, _ = zdot_model(nu=nu, z_term=z_term, threshold=z_threshold)
        m1_.fit(z_, u=x_t, x_dot=zd)

        try:
            # We could proceed with the initial z estimate, but since m1 is not perfect we try to compensate
            # by fitting a model for x_dot with z as the simulation of m1, given perfect knowledge of x_dot
            zs = m1_.simulate(np.array([z_[0, 0]]), t_t, u=x_t)
            zs = np.pad(zs.ravel(), (0, 1), mode='edge').reshape(-1, 1)
            m2_, _ = xdot_model(degree=1, threshold=x_threshold)
            m2_.fit(x_t, u=np.concatenate([y_t, zs, u_t], axis=1), x_dot=xd_t)
            m2_coef_ = m2_.coefficients().copy()

            # Using the model for simulation - here we combine the previous two models, reducing the controls
            # to u exclusively - notice that we simulate on the validation trajectory.
            # To recover the initialization of the hidden state we use the first model equation again,
            # with derivatives up to the second order of the output y on the first validation datapoint.
            z0 = np.dot(np.array([x_v[0, 0], y_v[0, 0], u_v[0, 0]]), m2_coef_[0, (0, 1, 3)]) - xd_v[0, 0]
            full_model = ode_model(u_v, t_v, x_grad=m2_, z_grad=m1_)

            xyz_qs = solve_ivp(full_model, [0.0, t_v[-1]], [x_v[0, 0], y_v[0, 0], z0], t_eval=t_v)['y']
            x_full_ = xyz_qs[0, :].reshape(-1, 1)
            y_full_ = xyz_qs[1, :].reshape(-1, 1)
            z_full_ = xyz_qs[2, :].reshape(-1, 1)

            # Computing the performance metrics on the validation trajectory
            z_mod_r2_ = r2_score(y_v.ravel(), y_full_.ravel())
            z_mod_rmse_ = np.sqrt(np.mean((y_v.ravel() - y_full_.ravel()) ** 2))

            if verbose:
                print('SINDy hidden model:')
                m1_.print()
                m2_.print()
                print("(y)' = 1.000 x")
                print('-' * 120)
                print(f'\tR2: {100 * z_mod_r2_:.3f}%, RMSE: {z_mod_rmse_:.3e}')

            # The process can be iterated; in this case, we need to recompute the z estimate
            # based on the parameters fitted in m2
            z_ = np.dot(np.concatenate([x_t, y_t, u_t], axis=1), m2_coef_[:, (0, 1, 3)].T) - xd_t

            # When using the process iteratively, we save the best model in terms of validation RMSE
            if z_mod_rmse is None or z_mod_rmse_ < z_mod_rmse:
                m1, m2, m2_coef = deepcopy(m1_), deepcopy(m2_), m2_coef_.copy()
                x_full, y_full, z_full = x_full_.copy(), y_full_.copy(), z_full_.copy()
                z_mod_rmse, z_mod_r2, z = z_mod_rmse_, z_mod_r2_, z_.copy()

        except Exception:
            warnings.warn(f'Iteration {it + 1} failed, terminating prematurely...')
            break

    return m1, m2, m2_coef, x_full, y_full, z_full, z_mod_rmse, z_mod_r2, z, z_est


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
            val_rmse = fit_x_model(
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

    m0, x_sindy, y_sindy, sindy_rmse, sindy_r2 = fit_x_model(
        t_t, y_t, x_t, u_t, xd_t,
        t_v, y_v, x_v, u_v, xd_v,
        verbose=verbose,
        **x_model_best_parameters
    )

    # 03 - Fitting the SINDy model for the hidden state
    def z_model_validation_rmse(params: dict):
        print(f'Parameters: {params}')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_rmse = fit_z_model(
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

    m1, m2, m2_coef, x_z_mod, y_z_mod, z_z_mod, z_mod_rmse, z_mod_r2, z, z_est = fit_z_model(
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
