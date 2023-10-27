"""Code based on https://pysindy.readthedocs.io/en/latest/examples/9_sindypi_with_sympy/example.html#Find-complex-PDE-with-SINDy-PI-with-PDE-functionality
"""
import contextlib
import warnings
import os

import hyperopt
import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt
import sympy as sp

from typing import Tuple
from fire import Fire
from hyperopt import hp, fmin, tpe
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
from src.utils.model_selection import *


def score(x_true, x_sim):
    r2 = r2_score(x_true.ravel(), x_sim.ravel())
    rmse = np.sqrt(np.mean((x_true.ravel() - x_sim.ravel()) ** 2))

    return r2, rmse


def get_model(
        model_type: str,
        order: int = 1,
        **model_kwargs
):
    if order == 1:
        feature_names = ['x', 'u']
    elif order == 2:
        feature_names = ['xd', 'x', 'u']
    else:
        raise ValueError(f'Possible valiues for "order" are 1 and 2!')

    optimizer = ps.STLSQ(**model_kwargs)

    if model_type == 'naive':
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(include_bias=False),
            feature_names=feature_names,
            optimizer=optimizer
        )
    elif model_type == 'sqrt':
        sqrt_library = ps.CustomLibrary(
            library_functions=[lambda x: np.sqrt(np.where(x >= 0.0, x, 0.0))],
            function_names=[lambda x: f'sqrt(max({x}, 0))']
        )
        poly_library = ps.PolynomialLibrary(include_bias=False)
        sindy_sqrt_library = ps.GeneralizedLibrary(libraries=[poly_library, sqrt_library])
        model = ps.SINDy(
            feature_names=feature_names,
            feature_library=sindy_sqrt_library,
            optimizer=optimizer
        )
    else:
        raise NotImplementedError(f'Unknown model type: "{model_type}"!')

    return model


def train_validate_model(
        t_test: np.ndarray,
        x_train: np.ndarray,
        u_train: np.ndarray,
        x_dot_train: np.ndarray,
        x_test: np.ndarray,
        u_test: np.ndarray,
        model_type: str,
        order: int = 1,
        x_ddot_train: np.ndarray = None,
        integrator_kws=None,
        **model_kwargs
):
    # -> instantiating the model
    if integrator_kws is None:
        integrator_kws = {}
    model = get_model(model_type, order=order, **model_kwargs)

    # -> fitting
    if order == 1:
        model.fit(x_train, u=u_train, x_dot=x_dot_train)
    elif order == 2:
        model.fit(
            np.concatenate([x_dot_train, x_train], axis=1),
            u=u_train,
            x_dot=np.concatenate([x_ddot_train, x_dot_train], axis=1)
        )
        # forcing the first state to be the derivative of the second
        model.model.steps[-1][1].coef_[1, :] = 0.0
        model.model.steps[-1][1].coef_[1, 0] = 1.0
    else:
        raise ValueError(f'Possible valiues for "order" are 1 and 2!')

    # -> simulating
    try:
        if order == 1:
            simulation = model.simulate(
                x_test[0, :], t=t_test, u=u_test, integrator_kws=integrator_kws
            )
            simulation = np.pad(simulation.ravel(), (0, 1), 'edge').reshape(-1, order)
        else:
            simulation = model.simulate(
                np.array([x_dot_train[0, 0], x_test[0, 0]]), t=t_test, u=u_test, integrator_kws=integrator_kws
            )
            simulation = np.pad(simulation, ((0, 1), (0, 0)), 'edge').reshape(-1, order)
    except Exception as e:
        warnings.warn(f'SINDy naive model failed ({type(e).__name__}: {e}), returning NaNs...')
        simulation = np.ones((t_test.shape[0], order)) * np.nan

    if simulation.shape[0] != t_test.shape[0]:
        simulation = np.pad(
            simulation,
            ((0, t_test.shape[0] - simulation.shape[0]), (0, 0)),
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
    # TODO: extract validation data to perform HPO
    train_data, validation_data, test_data, dt, t_train, t_val, t_test = load_data()

    # TODO: rescale back when saving the results
    # 01b - Rescaling the data (for numerical stability)
    train_data /= rescale_factor
    validation_data /= rescale_factor
    test_data /= rescale_factor

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

    # 03 - Training the models
    plt.figure()
    plt.plot(t_test, x_test, label='True')
    for readable_name, model_type, order, integrator_kws in zip(
            ['Naive SINDy', 'SINDy SQRT', 'SINDy 2nd order', 'SINDy 2nd order, SQRT'],
            ['naive', 'sqrt', 'naive', 'sqrt'],
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
        rstate = np.random.Generator(np.random.PCG64(seed=42))
        search_space = {
            'alpha': hp.loguniform('alpha', np.log(1e-3), np.log(1e3)),
            'threshold': hp.loguniform('threshold', np.log(1e-1), np.log(1e1))
        }

        best_parameters = hyperopt.space_eval(
            search_space,
            fmin(
                fn=validation_rmse,
                space=search_space,
                algo=tpe.suggest,
                max_evals=100,
                rstate=rstate
            )
        )

        # TODO: save outputs
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
        print(f'-> RMSE: {rmse:.4f}')

        plt.plot(t_test, simulation[:, -1], label=readable_name)



    plt.legend()
    plt.show()




if __name__ == '__main__':
    Fire(cascaded_tanks)
