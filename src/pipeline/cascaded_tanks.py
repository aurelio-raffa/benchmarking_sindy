"""Code based on https://pysindy.readthedocs.io/en/latest/examples/9_sindypi_with_sympy/example.html#Find-complex-PDE-with-SINDy-PI-with-PDE-functionality
"""
import warnings
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
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from __init__ import root_path
from src.utils.etl.cascaded_tanks import load_data
from src.utils.etl import prepare_data
from src.utils.symbolic_conversion import prepare_for_sympy
from src.utils.timeout import TimeoutException, TimeoutManager
from src.utils.preprocessing.compute_derivatives import compute_derivatives
from src.utils.simulation.wrappers import simulation_wrapper
from src.utils.functions import threshold
from src.utils.simulation.models.tank_simulator import tank_simulator


def score(x_true, x_sim):
    r2 = r2_score(x_true.ravel(), x_sim.ravel())
    rmse = np.sqrt(np.mean((x_true.ravel() - x_sim.ravel()) ** 2))

    print(f'-> R2: {100 * r2:.2f}%')
    print(f'-> RMSE: {rmse:.4f}')

    return r2, rmse


def get_model(
        model_type: str,
        order: int = 1,
):
    if order == 1:
        feature_names = ['x', 'u']
    elif order == 2:
        feature_names = ['xd', 'x', 'u']
    else:
        raise ValueError(f'Possible valiues for "order" are 1 and 2!')

    if model_type == 'naive':
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(),
            feature_names=feature_names
        )
    elif model_type == 'sqrt':
        sqrt_library = ps.CustomLibrary(
            library_functions=[lambda x: np.sqrt(np.where(x >= 0.0, x, 0.0))],
            function_names=[lambda x: f'sqrt(max({x}, 0))']
        )
        poly_library = ps.PolynomialLibrary()
        sindy_sqrt_library = ps.GeneralizedLibrary(libraries=[sqrt_library, poly_library])
        model = ps.SINDy(
            feature_names=feature_names,
            feature_library=sindy_sqrt_library
        )
    else:
        raise NotImplementedError(f'Unknown model type: "{model_type}"!')

    return model


def train_validate_model(
        t: np.ndarray,
        x_train: np.ndarray,
        u_train: np.ndarray,
        x_dot_train: np.ndarray,
        x_test: np.ndarray,
        u_test: np.ndarray,
        model_type: str,
        order: int = 1,
        x_ddot_train: np.ndarray = None,
        integrator_kws: dict = None,
):
    # -> instantiating the model
    model = get_model(model_type)

    # -> fitting
    if order == 1:
        model.fit(x_train, u=u_train, x_dot=x_dot_train)
    elif order == 2:
        model.fit(
            np.concatenate([x_dot_train, x_train], axis=1),
            u=u_train,
            x_dot=np.concatenate([x_ddot_train, x_dot_train], axis=1)
        )
    else:
        raise ValueError(f'Possible valiues for "order" are 1 and 2!')

    # -> simulating
    try:
        if order == 1:
            simulation = model.simulate(
                x_test[0, :], t=t, u=u_test, integrator_kws=integrator_kws
            )
            simulation = np.pad(simulation.ravel(), (0, 1), 'edge').reshape(-1, order)
        else:
            simulation = model.simulate(
                np.array([x_dot_train[0, 0], x_test[0, 0]]), t=t, u=u_test, integrator_kws=integrator_kws
            )
            simulation = np.pad(simulation, ((0, 1), (0, 0)), 'edge').reshape(-1, order)
    except Exception as e:
        warnings.warn(f'SINDy naive model failed ({type(e).__name__}: {e}), returning NaNs...')
        simulation = np.ones((t.shape[0], order)) * np.nan

    if simulation.shape[0] != t.shape[0]:
        simulation = np.pad(
            simulation,
            ((0, t.shape[0] - simulation.shape[0]), (0, 0)),
            mode='constant',
            constant_values=np.nan
        )

    # -> scoring
    try:
        r2, rmse = score(x_test, simulation[:, -1])
    except ValueError:
        r2 = -np.inf
        rmse = np.inf

    return model, simulation, r2, rmse


def cascaded_tanks(
        rescale_factor: float = 10.0,
        tvr_gamma: float = 8.0,
):
    # 00 - Helpers
    max_val = 10.0 / rescale_factor

    # 01 - Loading the data
    train_data, test_data, t, dt, tf = load_data()

    # TODO: rescale back when saving the results
    # 01b - Rescaling the data (for numerical stability)
    train_data /= rescale_factor
    test_data /= rescale_factor

    # 02 - Computing the derivatives (using TV regularization) and preparing the dataset
    x_train, u_train, x_dot_train, x_ddot_train = prepare_data(
        train_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2
    )
    x_test, u_test, x_dot_test, x_ddot_test = prepare_data(
        test_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=2
    )

    # 03 - Training the models
    # 03a - Naive SINDy
    print(' Naive SINDy '.center(120, '='))
    naive_sindy = ps.SINDy(
        feature_library=ps.PolynomialLibrary(),
        feature_names=['x', 'u']
    )

    # -> fitting
    naive_sindy.fit(x_train, u=u_train, x_dot=x_dot_train)

    # -> printing
    print('Equations:')
    naive_sindy.print()

    # -> simulating
    naive_sindy_sim = naive_sindy.simulate(x_test[0, :], t=t, u=u_test)
    naive_sindy_sim = np.pad(naive_sindy_sim.ravel(), (0, 1), 'edge')

    # -> scoring
    score(x_test, naive_sindy_sim)

    # 03b - SINDy with sqrt nonlinearities
    print(' SINDy, Square-root nonlinearities '.center(120, '='))
    sqrt_library = ps.CustomLibrary(
        library_functions=[
            lambda x: np.sqrt(np.where(x >= 0.0, x, 0.0))
        ],
        function_names=[
            lambda x: f'sqrt(max({x}, 0))'
        ]
    )
    poly_library = ps.PolynomialLibrary()
    sindy_sqrt_library = ps.GeneralizedLibrary(libraries=[sqrt_library, poly_library])
    sindy_sqrt = ps.SINDy(
        feature_names=['x', 'u'],
        feature_library=sindy_sqrt_library
    )

    # -> fitting
    sindy_sqrt.fit(x_train, u=u_train, x_dot=x_dot_train)

    # -> printing
    print('Equations:')
    sindy_sqrt.print()

    # -> simulating
    sindy_sqrt_sim = sindy_sqrt.simulate(x_test[0, :], t=t, u=u_test)
    sindy_sqrt_sim = np.pad(sindy_sqrt_sim.ravel(), (0, 1), 'edge')

    # -> scoring
    score(x_test, sindy_sqrt_sim)

    # 03c - SINDy second order
    print(' SINDy, second order '.center(120, '='))
    sindy_sqrt_library = ps.GeneralizedLibrary(libraries=[sqrt_library, poly_library])
    sindy_second_order = ps.SINDy(
        feature_library=sindy_sqrt_library,
        feature_names=['xd', 'x', 'u']
    )

    # -> fitting
    sindy_second_order.fit(
        np.concatenate([x_dot_train, x_train], axis=1),
        u=u_train,
        x_dot=np.concatenate([x_ddot_train, x_dot_train], axis=1)
    )

    # -> printing
    print('Equations:')
    sindy_second_order.print()

    # -> simulating
    sindy_second_order_sim = sindy_second_order.simulate(
        np.array([x_dot_train[0, 0], x_test[0, 0]]), t=t, u=u_test,
        integrator_kws={"method": "RK45"}
    )[1, :]
    sindy_second_order_sim = np.pad(sindy_second_order_sim.ravel(), (0, 1), 'edge')

    # -> scoring
    score(x_test, sindy_second_order_sim)

    # TODO: move to the end
    # -> plotting
    plt.figure()
    plt.plot(t, x_test, label='True')
    plt.plot(t, naive_sindy_sim, label='Naive SINDy')
    plt.plot(t, sindy_sqrt_sim, label='SINDy SQRT')
    plt.plot(t, sindy_second_order_sim, label='SINDy 2nd order')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    Fire(cascaded_tanks)
