"""Code based on https://pysindy.readthedocs.io/en/latest/examples/9_sindypi_with_sympy/example.html#Find-complex-PDE-with-SINDy-PI-with-PDE-functionality
"""
import os

import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt
import sympy as sp

from typing import Tuple
from fire import Fire
from scipy.integrate import solve_ivp
from pynumdiff.total_variation_regularization import jerk
from sklearn.metrics import r2_score

from __init__ import root_path
from src.utils.import_cascaded_tanks_data import import_cascaded_tanks_data

from src.utils.model_selection.cascaded_tanks import simulate_and_select
from src.utils.simulation.cascaded_tanks import prepare_for_simulation


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
        tvr_gamma: float = 10.0
):
    u = df['u'].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)

    # plt.figure()
    # plt.plot(y, label='y')
    # plt.plot(u, label='u')
    # plt.legend()
    # plt.show()
    x_dot, _ = compute_derivatives(y.ravel(), dt=dt, tvr_gamma=tvr_gamma)
    # x = np.concatenate(
    #     [
    #         x_ddot.reshape(-1, 1),
    #         x_dot.reshape(-1, 1),
    #         y.reshape(-1, 1)
    #     ],
    #     axis=1
    # )

    return y, u, x_dot.reshape(-1, 1)


rescale_factor: float = 10.0
max_val = 10.0 / rescale_factor


def double_thresh(y_):
    lwr_thrsh = np.where(y_ >= 0.0, y_, 0.0)
    res = np.where(lwr_thrsh <= max_val, lwr_thrsh, max_val)
    return res


def sim_hidden_state(
        t: np.ndarray,
        u: np.ndarray,
        tf: float,
        k1: float = 3.5,
        k3: float = 12,
        z0: float = 0.5
):
    def upper_tank_model(t_, x, eps=1e-6):
        x_dot = - k1 * np.sqrt(max(x, 0.0)) + k3 * np.interp(t_, t, u.ravel())
        if x < eps:
            return max(0.0, x_dot)
        elif x > max_val - eps:
            return min(0.0, x_dot)
        else:
            return x_dot

    z = solve_ivp(upper_tank_model, [0.0, tf], [z0], t_eval=t)['y'].T
    z = double_thresh(z).reshape(-1, 1)

    return z


# TODO: hpo
def main(
        tvr_gamma: float = 8.0,
        threshold: float = 0.00001,
        thresholder: str = "l1",
        tol: float = 1e-6,
        max_iter: int = 50000,
        precision: int = 10,
        simulation_timeout_seconds: int = 10,
):
    """Computes a second-order model with SINDy-PI for the cascaded tanks data.
    """
    # 00 - Variables and helpers

    # 01 - Loading the data
    train_data, test_data, t, dt, tf = import_cascaded_tanks_data()

    # TODO: rescale back when saving the results
    # 01b - Rescaling the data (for numerical stability)
    train_data /= rescale_factor
    test_data /= rescale_factor

    # 02 - Computing the derivatives (using TV regularization) and preparing the dataset
    x_train, u_train, x_dot_train = prepare_data(train_data, dt=dt, tvr_gamma=tvr_gamma)
    x_test, u_test, x_dot_test = prepare_data(test_data, dt=dt, tvr_gamma=tvr_gamma)

    # plt.plot(t, x_train, label='true')
    # plt.plot(t, np.cumsum(x_dot_train) * dt, label='est')
    # plt.legend()
    # plt.show()

    # 03 - Naive sindy with sqrt features
    # -> the features (naive SINDy) are:
    #    - x2
    #    - u
    #   for each one of these we must compute
    #    - identity
    #    - square
    #    - sqrt
    #    - pairwise product
    # #    - pairwise product with sqrt
    functions = [
        lambda x: x,  # identity for all terms
        # lambda x: x ** 2,
        lambda _, x, __, ___: np.sqrt(double_thresh(x)),
        lambda _, __, z, ___: np.sqrt(double_thresh(z)),
        lambda _, __, ___, u: np.sqrt(double_thresh(u)),
        # lambda x, y: x * y,  # pairwise product of all terms
        # lambda x, y: x * np.sqrt(double_thresh(y)),
        # lambda x, y: np.sqrt(double_thresh(x)) * y,
    ]
    feature_names = ['xd', 'x', 'z', 'u']
    function_names = [
        lambda x: x,  # identity for all terms
        # lambda x: x + "**2",
        lambda _, x, __, ___: f"sqrt(min(max({x}, 0), {max_val}))",
        lambda _, __, z, ___: f"sqrt(min(max({z}, 0), {max_val}))",
        lambda _, __, ___, u: f"sqrt(min(max({u}, 0), {max_val}))",
    ]

    lib = ps.PDELibrary(
        library_functions=functions,
        derivative_order=0,
        function_names=function_names,
        include_bias=False,
        implicit_terms=True,
        temporal_grid=t
    )

    # 04 - Fitting the model on the estimation data
    #   The SINDyPI solver will attempt to fit one implicit model for each candidate term.
    #   Beware that this could quickly get out of hand for large candidate libraries.
    sindy_opt = ps.SINDyPI(
        threshold=threshold,
        tol=tol,
        thresholder=thresholder,
        max_iter=max_iter,
    )
    model = ps.SINDy(
        optimizer=sindy_opt,
        feature_library=lib,
        feature_names=feature_names,
        differentiation_method=ps.FiniteDifference(drop_endpoints=True),
    )

    def objective_function(params):
        [k1_, k3_, z0_] = params

        z_ = sim_hidden_state(t, u_train, tf, k1=k1_, k3=k3_, z0=z0_)
        tfs = np.concatenate([x_dot_train, x_train, z_, u_train], axis=1)

        model.fit(tfs)

        score, simulation, model_equation = simulate_and_select(
            model,
            t,
            tf,
            z_,
            u_train,
            tfs,
            precision=precision,
            simulation_timeout_seconds=simulation_timeout_seconds
        )

        return score

    # [k1_opt, k3_opt, z0_opt] = minimize(objective_function, x0=np.array([3.5, 12.0, 0.5]), options={'maxiter': 10}).x
    #
    # print(k1_opt, k3_opt, z0_opt)
    # [k1_opt, k3_opt, z0_opt] = [19.0, 42.0, 0.6]
    [k1_opt, k3_opt, z0_opt] = [16.8, 41.5, 0.45]

    # 05 - Reconstruction of the unobserved state
    z_train = sim_hidden_state(t, u_train, tf, k1=k1_opt, k3=k3_opt, z0=z0_opt)

    training_feats = np.concatenate([x_dot_train, x_train, z_train, u_train], axis=1)
    model.fit(training_feats)
    model.print()

    score, simulation, model_equation = simulate_and_select(
        model,
        t,
        tf,
        z_train,
        u_train,
        training_feats,
        precision=precision,
        simulation_timeout_seconds=simulation_timeout_seconds
    )
    print(model_equation)
    print(f'final score_simulation: {score}')
    print(f'r2: {r2_score(x_train.ravel(), simulation.ravel())}')

    plt.plot(t, x_train, label='true')
    plt.plot(t, z_train, label='hidden')
    plt.plot(t, simulation, label='sim')
    plt.legend()
    plt.show()

    test_initial_conitions = x_test[0, 0:1].copy()
    test_y_true = x_test.copy().ravel()

    z_test = sim_hidden_state(t, u_test, tf, k1=k1_opt, k3=k3_opt, z0=z0_opt)

    ode_fun = sp.lambdify([sp.symbols('x'), sp.symbols('z'), sp.symbols('u')], model_equation)

    # We prepare to run the simulation
    test_ode = prepare_for_simulation(ode_fun, t=t, z=z_test, u=u_test)

    y_out = solve_ivp(test_ode, [0.0, tf], test_initial_conitions, t_eval=t)['y'].T
    print(y_out.shape)
    print(y_out)
    # test_feats = np.concatenate([x_dot_train, x_train, z_train, u_train], axis=1)
    test_rmse = rescale_factor * np.sqrt(np.mean((y_out - test_y_true) ** 2))
    test_r2 = r2_score(test_y_true, y_out)

    print(f'Test RMSE: {test_rmse:.3f} | Test R2: {100 * test_r2:.3f}%')

    benchmark_data = pd.read_csv(os.path.join(root_path, 'data/Benchmarks/cascaded_tanks.csv'))

    # print(f'ARX test R2: {r2_score(test_y_true, benchmark_data["ARX"]):.3f}')

    plt.figure()
    plt.plot(t, rescale_factor * test_y_true, label='true', color='black', linewidth=2)
    plt.plot(t, rescale_factor * y_out, label=f'simulation')
    plt.plot(t, benchmark_data['ARX'], label='ARX')
    plt.plot(t, benchmark_data['SINDy_prior'], label='SINDy_prior')
    plt.plot(t, benchmark_data['SINDy_naif'], label='SINDy_naif')
    # plt.ylim([-rescale_factor * 0.2, rescale_factor * 1.2])
    plt.legend()
    plt.show()

    results = pd.DataFrame(
        np.concatenate([t.reshape(-1, 1), rescale_factor * y_out.reshape(-1, 1), rescale_factor * x_test], axis=1),
        columns=['time', 'x', 'true']
    )
    print(results)
    results.to_csv(os.path.join(root_path, 'outputs/SINDy_good_library.csv'), index=False)


if __name__ == '__main__':
    Fire(main)
