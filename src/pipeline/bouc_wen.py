import os

import pysindy as ps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Any
from fire import Fire
from pynumdiff.total_variation_regularization import jerk
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score


from __init__ import root_path
from src.utils.etl import prepare_data
from src.utils.etl.bouc_wen import load_data


def ode_model(u: np.ndarray, t: np.ndarray, x_grad: Any, z_grad: Any):
    # preparing the model for simulation
    def mod(t_, xyz_):
        [x_, y_, z_] = xyz_
        u_ = np.interp(t_, t, u.ravel())

        zd_ = z_grad.predict(np.array([[z_]]), u=np.array([[x_]])).ravel()[0]
        # xd_ = (u_ - k * y_ - c * x_ - z_) / m
        xd_ = x_grad.predict(np.array([[x_]]), u=np.array([[y_, z_, u_]])).ravel()[0]
        yd_ = x_

        return np.array([xd_, yd_, zd_])

    return mod


def xdot_noz_model():
    """simple library consisting of a linear combination of inputs and state
    """
    feature_names = ['x', 'y', 'u']
    lib = ps.PolynomialLibrary(
        include_bias=False
    )
    optimizer = ps.STLSQ(threshold=0.00001, alpha=10.0)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        feature_names=feature_names,
        differentiation_method=None,
    )
    return model, feature_names


def xdot_model():
    """simple library consisting of a linear combination of inputs and state
    """
    feature_names = ['x', 'y', 'z', 'u']
    function_names = functions = [lambda x: x]
    lib = ps.CustomLibrary(
        library_functions=functions,
        function_names=function_names,
        include_bias=False
    )
    optimizer = ps.STLSQ(threshold=0.00001, alpha=10.0)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        feature_names=feature_names,
        differentiation_method=None,
    )
    return model, feature_names


def zdot_simplified_model():
    """library consisting of transformations akin the Bouc-Wen model
    """
    feature_names = ['z', 'x']
    functions = [
        lambda _, x: x,     # TODO: vedi cosa succede con lambda x: x...
        # lambda x: x,  # TODO: vedi cosa succede con lambda x: x...
        lambda z, x: z * np.abs(x),
        lambda z, x: np.abs(z) * x
    ]
    function_names = [
        lambda _, x: x,
        # lambda x: x,
        lambda z, x: f'{z} * |{x}|',
        lambda z, x: f'|{z}| * {x}'
    ]
    lib = ps.CustomLibrary(
        library_functions=functions,
        function_names=function_names,
        include_bias=False
    )
    optimizer = ps.STLSQ(threshold=1.0)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        feature_names=feature_names,
        differentiation_method=None,
    )
    return model, feature_names


def zdot_model(nu: float = 1.0):
    """library consisting of transformations akin the Bouc-Wen model
    """
    feature_names = ['z', 'x', 'y', 'u']
    functions = [
        lambda x: x,
        lambda z, x, _, __: z * x,
        lambda z, x, _, __: z * np.abs(x),
        lambda z, x, _, __: np.power(np.abs(z), nu) * x,
        lambda z, x, _, __: np.power(np.abs(z), nu) * np.abs(x),
        lambda z, x, _, __: np.power(np.abs(z), nu) * np.abs(x) * np.sign(z),
    ]
    function_names = [
        lambda x: x,
        lambda z, x, _, __: f'{z} * {x}',
        lambda z, x, _, __: f'{z} * |{x}|',
        lambda z, x, _, __: f'|{z}| ** nu * {x}',
        lambda z, x, _, __: f'|{z}| ** nu * |{x}|',
        lambda z, x, _, __: f'|{z}| ** nu * |{x}| * sign({z})',
    ]
    lib = ps.CustomLibrary(
        library_functions=functions,
        function_names=function_names,
        include_bias=False
    )
    optimizer = ps.STLSQ(threshold=0.1, alpha=1000.0)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        feature_names=feature_names,
        differentiation_method=None,
    )
    return model, feature_names

def true_z_model(
        x: np.ndarray,
        t: np.ndarray,
        alpha: float = 5e4,
        beta: float = 1e3,
        gamma: float = 0.8,
        delta: float = -1.1,
        nu: float = 1.0,
):
    x_c = x.copy().ravel()

    def grad_z(t_, z):
        xt = np.interp(t_, t, x_c)
        if nu == 1.0:
            zd = alpha * xt - beta * gamma * np.abs(xt) * z - beta * delta * xt * np.abs(z)
        else:
            assert nu > 1.0, 'nu must be greater than 1.0'
            z_abs_nu_m1_z = np.power(np.abs(z), nu - 1) * z
            z_abs_nu = np.power(np.abs(z), nu)
            zd = alpha * xt - beta * gamma * np.abs(xt) * z_abs_nu_m1_z - beta * delta * xt * z_abs_nu

        return np.array([zd])

    return grad_z

def bouc_wen(
        output_path: str = None,
        dt: float = 1 / 750.0,
        tvr_gamma: float = 0.000001,
        threshold: float = 0.00001,
        thresholder: str = "l1",
        tol: float = 1e-6,
        max_iter: int = 50000,
        training_samples: int = 1000,
        validation_samples: int = 1000,
        alternating_iterations: int = 1,
        verbose: bool = True,
):
    # TODO: extract validation data
    training_data, validation_data, test1_data, test2_data = load_data(
        training_samples=training_samples,
        validation_samples=validation_samples,
    )
    y_t, u_t, x_t, xd_t = prepare_data(training_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2)
    y_v, u_v, x_v, xd_v = prepare_data(validation_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2)
    t = np.arange(0, y_t.shape[0]) * dt

    # initial parameters and hidden state guess
    m, k, c = 2.0, 5e4, 10.0
    z = u_t - k * y_t - c * x_t - m * xd_t  # ==> xd = (u - k * y - c * x - z) / m

    # # simulating the linear model
    # def linear_model(t_, xy_):
    #     [x_, y_] = xy_
    #     u_ = np.interp(t_, t, u.ravel())
    #
    #     xd_ = (u_ - k * y_ - c * x_) / m
    #     yd_ = x_
    #
    #     return np.array([xd_, yd_])
    #
    # # zxy_lin = solve_ivp(linear_model, [0.0, t[-1]], [x[0, 0], y[0, 0]], t_eval=t)['y']
    # # x_lin = zxy_lin[0, :].reshape(-1, 1)
    # # y_lin = zxy_lin[1, :].reshape(-1, 1)

    m0, _ = xdot_noz_model()
    m0.fit(np.concatenate([x_t, y_t], axis=1), u=u_t, x_dot=np.concatenate([xd_t, x_t], axis=1))

    if verbose:
        print('SINDy base model:')
        m0.print()
    xy_sindy = m0.simulate(np.array([x_v[0, 0], y_v[0, 0]]), t, u=u_v)
    x_sindy = np.pad(xy_sindy[:, 0], (0, 1), mode='edge').reshape(-1, 1)
    y_sindy = np.pad(xy_sindy[:, 1], (0, 1), mode='edge').reshape(-1, 1)

    upsample_factor = 20
    grad_z = true_z_model(x_v, t)
    z_hf = solve_ivp(grad_z, [0.0, t[-1]], [0.0], t_eval=np.linspace(0.0, t[-1], upsample_factor * t.shape[0]))['y']
    z_hf = z_hf.ravel()[::upsample_factor].reshape(-1, 1)

    # TODO: validate and save best at each iteration
    # fitting the model for the hidden state
    zc = z.copy()
    z_full = x_full = y_full = None
    m1 = m2 = None
    m2_coef = None
    for it in range(alternating_iterations):
        if verbose:
            print(f' Iteration {it+1} '.center(120, '='))

        zd = jerk(zc.ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1)
        m1, _ = zdot_simplified_model()
        m1.fit(zc, u=x_t, x_dot=zd)

        zs = m1.simulate(np.array([zc[0, 0]]), t, u=x_t)
        zs = np.pad(zs.ravel(), (0, 1), mode='edge').reshape(-1, 1)
        m2, _ = xdot_model()
        m2.fit(x_t, u=np.concatenate([y_t, zs, u_t], axis=1), x_dot=xd_t)
        m2_coef = m2.coefficients().copy()

        zc = -1.0 * (xd_t - np.dot(np.concatenate([x_t, y_t, u_t], axis=1), m2_coef[:, (0, 1, 3)].T))

        if verbose:
            print('-' * 120)
            print('SINDy z model:')
            m1.print()
            m2.print()
            print("(y)' = 1.000 x")

    # preparing the model for simulation
    full_model = ode_model(u_v, t, x_grad=m2, z_grad=m1)

    # simulating
    xyz_qs = solve_ivp(full_model, [0.0, t[-1]], [x_v[0, 0], y_v[0, 0], 0.0], t_eval=t)['y']
    x_full = xyz_qs[0, :].reshape(-1, 1)
    y_full = xyz_qs[1, :].reshape(-1, 1)
    z_full = xyz_qs[2, :].reshape(-1, 1)

    # TODO: test on valid
    if verbose:
        print('=' * 120)
        print(f'SINDy base model:')
        print(f'\tR2: {100 * r2_score(y_v.ravel(), y_sindy.ravel()):.3f}%, RMSE: {np.sqrt(np.mean((y_v.ravel() - y_sindy.ravel()) ** 2)):.3e}')
        print(f'SINDy z model:')
        print(f'\tR2: {100 * r2_score(y_v.ravel(), y_full.ravel()):.3f}%, RMSE: {np.sqrt(np.mean((y_v.ravel() - y_full.ravel()) ** 2)):.3e}')

    # TODO: run model on test data and save metrics!

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    ax1.plot(t, z, label='z (estimated)')
    ax1.plot(t, z_full, label='z (simulated)')
    ax1.plot(t, z_hf, label='z (high fidelity)')
    ax1.plot(t, zc, label='z (c)')
    ax1.legend()

    ax2.plot(t, x_v, label='x', zorder=4)
    ax2.plot(t, x_full, label='x (simulated)', zorder=3)
    ax2.plot(t, x_sindy, label='x (sindy)', zorder=2)
    ax2.legend()

    ax3.plot(t, y_v, label='y', zorder=4)
    ax3.plot(t, y_full, label='y (simulated)', zorder=3)
    ax3.plot(t, y_sindy, label='y (sindy)', zorder=2)
    ax3.legend()

    plt.show()

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
