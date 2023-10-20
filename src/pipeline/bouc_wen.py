import os

import pysindy as ps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from fire import Fire
from pynumdiff.total_variation_regularization import jerk
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score

from __init__ import root_path


# TODO: remove code duplication
def compute_derivatives(
        x: np.ndarray,
        dt: float,
        tvr_gamma: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the second derivative of the data x using TV regularization.
    """
    x_hat, x_dot = jerk(x, dt, params=[tvr_gamma])
    _, x_ddot = jerk(x_dot, dt, params=[tvr_gamma])
    return x_dot, x_ddot


def prepare_data(
        df: pd.DataFrame,
        dt: float,
        tvr_gamma: float = 1.0
):
    u = df['u'].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)

    x, xd = compute_derivatives(y.ravel(), dt=dt, tvr_gamma=tvr_gamma)

    return y, u, x.reshape(-1, 1), xd.reshape(-1, 1)


def xdot_noz_model():
    """simple library consisting of a linear combination of inputs and state
    """
    feature_names = ['x', 'y', 'u']
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


def xdot_model():
    """simple library consisting of a linear combination of inputs and state
    """
    feature_names = ['x', 'y', 'u', 'z']
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
        lambda _, x: x, # TODO: vedi cosa succede con lambda x: x...
        # lambda z, x: z * x,
        lambda z, x: z * np.abs(x),
        lambda z, x: np.abs(z) * x,
        # lambda z, x: np.abs(z) * np.abs(x),
        # lambda z, x: z * np.abs(x),
    ]
    function_names = [
        lambda _, x: x,
        # lambda z, x: f'{z} * {x}',
        lambda z, x: f'{z} * |{x}|',
        lambda z, x: f'|{z}| * {x}',
        # lambda z, x: f'|{z}| * |{x}|',
        # lambda z, x: f'{z} * |{x}|',
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
        input_path: str = 'data/BoucWen/Train signals/train_data.csv',
        output_path: str = None,
        dt: float = 1 / 750.0,
        tvr_gamma: float = 0.000001,
        threshold: float = 0.00001,
        thresholder: str = "l1",
        tol: float = 1e-6,
        max_iter: int = 50000,
):
    data = pd.read_csv(os.path.join(root_path, input_path))
    # TODO: remove
    data = data.iloc[:2000, :].copy()

    y, u, x, xd = prepare_data(data, dt=dt, tvr_gamma=tvr_gamma)
    t = np.arange(0, y.shape[0]) * dt
    # z = np.zeros_like(y)

    m, k, c = 2.0, 5e4, 10.0
    z = u - k * y - c * x - m * xd
    for i in range(10):
        # m0, _ = xdot_noz_model()
        # m0.fit(
        #     x,
        #     u=np.concatenate([y, u], axis=1),
        #     x_dot=xd
        # )
        # m0.print()
        # x_sim = m0.simulate(t=t, x0=x[0, :], u=np.concatenate([y, u], axis=1))
        # x_sim = np.pad(x_sim.ravel(), (0, 1), mode='edge').reshape(-1, 1)
        # x_sim_dot = jerk(x_sim.ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1)
        # z_init = x_sim_dot - xd
        # z_init_dot = jerk(z_init.ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1)
        # z_pred = m0.predict(x, u=np.concatenate([y, u], axis=1)).reshape(-1, 1) - xd
        # z_pred_dot = jerk(z_pred.ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1)

        # zh, zd = jerk(z.ravel(), dt, params=[tvr_gamma])
        # zh = zh.reshape(-1, 1)
        # zd = zd.reshape(-1, 1)
        zd = jerk(z.ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1)

        # grad_z = true_z_model(x, t)
        # z_sol = solve_ivp(grad_z, [0.0, t[-1]], z[0, :], t_eval=np.linspace(0.0, t[-1], 20 * t.shape[0]))['y']
        # z_sol = z_sol.ravel()[::20].reshape(-1, 1)
        # # z_sol_dot = jerk(z_sol.ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1)

        m1, _ = zdot_simplified_model()
        m1.fit(
            z,
            # z_sol,
            u=x,
            # x_dot=z_sol_dot
            x_dot=zd
        )
        # m1.fit(
        #     # z_init,
        #     z_pred,
        #     u=x,
        #     x_dot=z_pred_dot
        #     # x_dot=z_init_dot
        # )
        m1.print()

        z_new = np.pad(m1.simulate(t=t, x0=z[0, :], u=x).ravel(), (0, 1), mode='edge').reshape(-1, 1)

        # _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        #
        # ax1.plot(t, z, label='z')
        # ax1.plot(t, zh, label='zh')
        # ax1.plot(t, z_new, label='z_new')
        # ax1.plot(t, z_sol, label='z_sol')
        # # ax1.plot(t, z_init, label='z_init')
        # ax1.plot(t, z_pred, label='z_pred')
        # ax1.legend()
        #
        # ax2.plot(t, zd, label='zd')
        # ax2.plot(t, z_sol_dot, label='z_sol_dot')
        # ax2.legend()
        #
        # plt.show()
        # # exit()


        m2, _ = xdot_model()
        m2.fit(
            np.concatenate([x, y], axis=1),
            # WARNING
            # u=np.concatenate([u, z_new], axis=1),
            u=np.concatenate([u, z], axis=1),
            x_dot=np.concatenate([xd, x], axis=1),
        )
        m2.print()

        # WARNING
        # sindy_sol = m2.simulate(t=t, x0=np.array([x[0, 0], y[0, 0]]), u=np.concatenate([u, z_new], axis=1))
        sindy_sol = m2.simulate(t=t, x0=np.array([x[0, 0], y[0, 0]]), u=np.concatenate([u, z], axis=1))
        x_sindy = np.pad(sindy_sol[:, 0], (0, 1), mode='edge').reshape(-1, 1)
        y_sindy = np.pad(sindy_sol[:, 1], (0, 1), mode='edge').reshape(-1, 1)

        # TODO: simulate full model in open loop now...
        # TODO: remove hardcoding
        # TODO: automate eq. extraction
        def final_model_lambdax(t_, zxy_):
            # (z)' = -18.215 z + 48243.385 x + -429.901 z * |x| + 593.530 |z| * x
            # (x)' = -0.323 x + -21251.107 y + 0.400 u + -0.644 z
            # (y)' = 1.000 x
            [z_, x_, y_] = zxy_
            u_ = np.interp(t_, t, u.ravel())

            zd_ = -18.215 * z_ + 48243.385 * x_ - 429.901 * z_ * np.abs(x_) + 593.530 * np.abs(z_) * x_
            xd_ = -0.323 * x_ - 21251.107 * y_ + 0.400 * u_ + -0.644 * z_
            yd_ = 1.000 * x_

            return np.array([zd_, xd_, yd_])

        def final_model_narrow(t_, zxy_):
            # (z)' = 47965.141 x + -538.060 z * |x| + 621.801 |z| * x
            # (x)' = -6.452 x + -24279.075 y + 0.403 u + -0.585 z
            # (y)' = 1.000 x
            [z_, x_, y_] = zxy_
            u_ = np.interp(t_, t, u.ravel())

            # TODO: change with model prediction from m1
            zd_ = 47965.141 * x_ - 538.060 * z_ * np.abs(x_) + 621.801 * np.abs(z_) * x_
            # xd_ = -6.452 * x_ - 24279.075 * y_ + 0.403 * u_ + -0.585 * z_
            # yd_ = 1.000 * x_

            # (x)' = -5.000 x + -25000.000 y + 0.500 u + -0.500 z
            # (y)' = 1.000 x
            xd_ = -5.000 * x_ - 25000.000 * y_ + 0.500 * u_ + -0.500 * z_
            yd_ = 1.000 * x_

            return np.array([zd_, xd_, yd_])

        # zxy_sol = solve_ivp(final_model_lambdax, [0.0, t[-1]], [z[0, 0], x[0, 0], y[0, 0]], t_eval=t)['y']
        zxy_sol = solve_ivp(final_model_narrow, [0.0, t[-1]], [z[0, 0], x[0, 0], y[0, 0]], t_eval=t)['y']

        z_final = zxy_sol[0, :].reshape(-1, 1)
        x_final = zxy_sol[1, :].reshape(-1, 1)
        y_final = zxy_sol[2, :].reshape(-1, 1)

        print(f'Simulation score: {100 * r2_score(y.ravel(), y_final.ravel()):.3f}%')

        # _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
        _, (ax1, ax3, ax4) = plt.subplots(3, 1, sharex=True)

        ax1.plot(t, z, label='z')
        # ax1.plot(t, zh, label='zh')
        ax1.plot(t, z_new, label='z_new')
        ax1.plot(t, z_final, label='z_fmod')
        # ax1.plot(t, z_sol, label='z_sol')
        ax1.legend()

        # ax2.plot(t, zd, label='zd')
        # ax2.plot(t, z_sol_dot, label='z_sol_dot')
        # ax2.legend()

        ax3.plot(t, x, label='x')
        ax3.plot(t, x_sindy, label='x_sindy')
        ax3.plot(t, x_final, label='x_fmod')
        ax3.legend()

        ax4.plot(t, y, label='y')
        ax4.plot(t, y_sindy, label='y_sindy')
        ax4.plot(t, y_final, label='y_fmod')
        ax4.legend()

        plt.show()

        # ax = plt.figure().add_subplot(projection='3d')
        # ax.plot(x.ravel(), y.ravel(), z.ravel(), label='original data')
        # ax.plot(x_sindy.ravel(), y_sindy.ravel(), z_new.ravel(), label='sindy')
        # ax.plot(x_final.ravel(), y_final.ravel(), z_final.ravel(), label='final model')
        # ax.legend()
        # plt.show()
        #
        # exit()
        # c = m1.coefficients()[0, 1]
        # k = m1.coefficients()[0, 2]
        # m = 1.0 / m1.coefficients()[0, 3]
        # c /= m
        # k /= m



    # m1, _ = xdot_model()
    # m1.fit(x, u=np.concatenate([y, u], axis=1), x_dot=xd)
    # m1.print()
    #
    # z = xd - np.dot(np.concatenate([x, y, u], axis=1), m1.coefficients().T)
    # zd = jerk(z.ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1)
    #
    # m2, _ = zdot_model(nu=1.0)
    # m2.fit(z, u=x, x_dot=zd)
    # m2.print()
    # z = m2.simulate(t=t, x0=[0.0], u=x)
    # z = np.pad(z.ravel(), (0, 1), mode='edge').reshape(-1, 1)
    # print(z)
    # assert z.shape == z.shape
    #
    # m3, _ = xdot_model()
    # m3.fit(x, u=np.concatenate([y, u, z], axis=1), x_dot=xd)
    # m3.print()


    # training_feats = np.concatenate([x, y], axis=1)
    # model.fit(training_feats, u=u, x_dot=xd)
    # model.print()
    #
    # # xd = -0.002 xdd + -112.039 x + 0.001 u
    # z = x + 0.002 * xd + 112.039 * y - 0.001 * u
    # z, z_dot = jerk(z.ravel(), dt, params=[tvr_gamma])
    #
    #
    # # training_feats = np.concatenate([x_dot, z.reshape(-1, 1)], axis=1)
    # model.fit(z.reshape(-1, 1), u=x, x_dot=z_dot)
    # model.print()



if __name__ == '__main__':
    Fire(bouc_wen)
