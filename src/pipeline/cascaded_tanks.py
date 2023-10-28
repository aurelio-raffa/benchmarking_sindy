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
from sympy import solve
from tqdm import tqdm

from __init__ import root_path
from src.utils.etl.cascaded_tanks import load_data
from src.utils.etl import prepare_data
from src.utils.model_selection.cascaded_tanks import simulate_and_select
from src.utils.simulation.cascaded_tanks import prepare_for_simulation
from src.utils.symbolic_conversion import prepare_for_sympy
from src.utils.timeout import TimeoutException, TimeoutManager
from src.utils.preprocessing.compute_derivatives import compute_derivatives
from src.utils.simulation.wrappers import simulation_wrapper
from src.utils.functions import threshold
from src.utils.simulation.models.tank_simulator import tank_simulator
from src.utils.model_selection import *


def score(x_true, x_sim):
    try:
        r2 = r2_score(x_true.ravel(), x_sim.ravel())
        rmse = np.sqrt(np.mean((x_true.ravel() - x_sim.ravel()) ** 2))
    except ValueError:
        r2 = -np.inf
        rmse = np.inf

    return r2, rmse


def pad_simulation(simulation, t_test):
    if simulation.shape[0] != t_test.shape[0]:
        simulation = np.pad(
            simulation,
            ((0, t_test.shape[0] - simulation.shape[0]), (0, 0)),
            mode='constant',
            constant_values=np.nan
        )

    return simulation


def implicit_model(
        t_train: np.ndarray,
        **model_kwargs
):
    functions = [
        lambda x: x,  # identity for all terms
        # lambda x: x ** 2,
        lambda _, x, __, ___: np.sqrt(threshold(x)),
        lambda _, __, z, ___: np.sqrt(threshold(z)),
        lambda _, __, ___, u: np.sqrt(threshold(u)),
        # lambda x, y: x * y,  # pairwise product of all terms
        # lambda x, y: x * np.sqrt(double_thresh(y)),
        # lambda x, y: np.sqrt(double_thresh(x)) * y,
    ]
    feature_names = ['xd', 'x', 'z', 'u']
    function_names = [
        lambda x: x,  # identity for all terms
        # lambda x: x + "**2",
        lambda _, x, __, ___: f"sqrt({x})",
        lambda _, __, z, ___: f"sqrt({z})",
        lambda _, __, ___, u: f"sqrt({u})",
    ]
    lib = ps.PDELibrary(
        library_functions=functions,
        derivative_order=0,
        function_names=function_names,
        include_bias=False,
        implicit_terms=True,
        temporal_grid=t_train,
    )

    # 04 - Fitting the model on the estimation data
    #   The SINDyPI solver will attempt to fit one implicit model for each candidate term.
    #   Beware that this could quickly get out of hand for large candidate libraries.
    sindy_opt = ps.SINDyPI(
        **model_kwargs
    )
    model = ps.SINDy(
        optimizer=sindy_opt,
        feature_library=lib,
        feature_names=feature_names,
        differentiation_method=ps.FiniteDifference(drop_endpoints=True),
    )

    return model


def get_model(
        model_type: str,
        order: int = 1,
        z: bool = False,
        u: bool = True,
        **model_kwargs
):
    if order == 1:
        feature_names = ['x']
    elif order == 2:
        feature_names = ['xd', 'x']
    else:
        raise ValueError(f'Possible valiues for "order" are 1 and 2!')

    if z:
        feature_names.append('z')
    if u:
        feature_names.append('u')

    optimizer = ps.STLSQ(**model_kwargs)

    if model_type == 'naive':
        model = ps.SINDy(
            feature_library=ps.PolynomialLibrary(include_bias=False),
            feature_names=feature_names,
            optimizer=optimizer
        )
    elif model_type == 'sqrt':
        sqrt_library = ps.CustomLibrary(
            library_functions=[lambda x: np.sqrt(threshold(x))],
            function_names=[lambda x: f'sqrt({x})']
        )
        # if not z:
        #     linear_library = ps.PolynomialLibrary(include_bias=False, degree=1)
        # elif u:
        #     assert order == 1
        #     linear_library = ps.CustomLibrary(
        #         library_functions=[
        #             lambda x, _, u_: x,
        #             lambda x, _, u_: u,
        #         ],
        #         function_names=[
        #             lambda x, _, u_: x,
        #             lambda x, _, u_: u,
        #         ]
        #     )
        # else:
        #     assert order == 1
        #     linear_library = ps.CustomLibrary(
        #         library_functions=[
        #             lambda x, _: x,
        #         ],
        #         function_names=[
        #             lambda x, _: x,
        #         ]
        #     )
        linear_library = ps.PolynomialLibrary(include_bias=False, degree=1)
        sindy_sqrt_library = ps.GeneralizedLibrary(libraries=[linear_library, sqrt_library])
        model = ps.SINDy(
            feature_names=feature_names,
            feature_library=sindy_sqrt_library,
            optimizer=optimizer
        )
    elif model_type == 'sqrt_poly':
        sqrt_library = ps.CustomLibrary(
            library_functions=[lambda x: np.sqrt(threshold(x))],
            function_names=[lambda x: f'sqrt({x})']
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

    # -> scoring
    simulation = pad_simulation(simulation, t_test)
    r2, rmse = score(x_test, simulation[:, -1])

    return model, simulation, r2, rmse


def train_validate_hidden_model(
        t_train: np.ndarray,
        t_test: np.ndarray,
        x_train: np.ndarray,
        u_train: np.ndarray,
        x_dot_train: np.ndarray,
        x_test: np.ndarray,
        u_test: np.ndarray,
        x_dot_test: np.ndarray,
        model_type: str,
        k1: float = 3.5,
        k3: float = 12,
        z0: float = 0.5,
        precision: int = 10,
        implicit: bool = False,
        integrator_kws=None,
        simulation_timeout_seconds: int = 30,
        **model_kwargs
):
    # -> instantiating the model
    if integrator_kws is None:
        integrator_kws = {}
    if implicit:
        model = implicit_model(t_train, **model_kwargs)
    else:
        model = get_model(model_type, z=True, u=False, order=1, **model_kwargs)

    # step 1: simulating the hidden state
    z_train = tank_simulator(
        t=t_train,
        u=u_train,
        tf=t_train[-1],
        k1=k1,
        k3=k3,
        z0=z0
    )

    # -> fitting
    if implicit:
        training_feats = np.concatenate([x_dot_train, x_train, z_train, u_train], axis=1)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            model.fit(training_feats, quiet=True)

        train_score, train_simulation, model_equation = simulate_and_select(
            model,
            t_train,
            t_train[-1],
            z_train,
            u_train,
            training_feats,
            precision=precision,
            simulation_timeout_seconds=simulation_timeout_seconds
        )

        # -> simulating
        test_initial_conitions = x_test[0, 0:1].copy()
        z_sim = tank_simulator(t_test, u_test, t_test[-1], k1=k1, k3=k3, z0=0.0)

        ode_fun = sp.lambdify([sp.symbols('x'), sp.symbols('z'), sp.symbols('u')], model_equation)
        test_ode = prepare_for_simulation(ode_fun, t=t_test, z=z_sim, u=u_test)
        simulation = solve_ivp(test_ode, [0.0, t_test[-1]], test_initial_conitions, t_eval=t_test)['y'].T

    else:
        # model.fit(x_train, u=np.concatenate([z_train, u_train], axis=1), x_dot=x_dot_train)

        model.fit(x_train, u=z_train, x_dot=x_dot_train)

        model.print()
        # rhs = prepare_for_sympy(model.equations(precision=precision)[0])
        # sol = sp.solve(sp.Eq(sp.sympify('xd'), sp.sympify(rhs)), sp.symbols('z'))
        # fun = sp.lambdify([sp.symbols('xd'), sp.symbols('x'), sp.symbols('u')], sol)
        # z0s = fun(x_dot_test[0, 0], x_test[0, 0], u_test[0, 0])
        #
        # z0_ = [z_ for z_ in z0s if 0 < z_ < 1]
        # if len(z0_) == 0:
        #     print(f'No valid initial conditions found, using {z0} instead')
        #     z0_ = z0
        # elif len(z0_) == 1:
        #     z0_ = z0_[0]
        #
        #     print(z0_)
        # else:
        #     print(f'Multiple valid initial conditions found, using the mean: {z0_}')
        #     z0_ = np.mean(z0_)
        z0_ = 0.0

        # -> simulating
        try:
            z_sim = tank_simulator(
                t=t_test,
                u=u_test,
                tf=t_test[-1],
                k1=k1,
                k3=k3,
                z0=z0_
            )
            # simulation = model.simulate(
            #     x_test[0, :], t=t_test, u=np.concatenate([z_sim, u_test], axis=1), integrator_kws=integrator_kws
            # )
            simulation = model.simulate(
                x_test[0, :], t=t_test, u=z_sim, integrator_kws=integrator_kws
            )
            simulation = np.pad(simulation.ravel(), (0, 1), 'edge').reshape(-1, 1)
        except Exception as e:
            warnings.warn(f'SINDy naive model failed ({type(e).__name__}: {e}), returning NaNs...')
            simulation = np.ones((t_test.shape[0], 1)) * np.nan

    # -> scoring
    simulation = pad_simulation(simulation, t_test)
    r2, rmse = score(x_test, simulation[:, -1])

    # plt.figure()
    # plt.plot(t_test, x_test)
    # plt.plot(t_test, z_sim)
    # plt.plot(t_test, simulation)
    # plt.show()

    return model, simulation, r2, rmse


def cascaded_tanks(
        rescale_factor: float = 10.0,
        tvr_gamma: float = 1.0,
):
    # 01 - Loading the data
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

    # 03a - Training the non-informed models
    plt.figure()
    plt.plot(t_test, x_test, label='True')
    # for readable_name, model_type, order, integrator_kws in zip(
    #         ['Naive SINDy', 'SINDy SQRT', 'SINDy 2nd order', 'SINDy 2nd order, SQRT'],
    #         ['naive', 'sqrt_poly', 'naive', 'sqrt_poly'],
    #         [1, 1, 2, 2],
    #         [{}, {}, {'method': 'Radau'}, {'method': 'Radau'}]
    # ):
    #     print(f' {readable_name} '.center(120, '='))
    #
    #     def validation_rmse(params: dict):
    #         with warnings.catch_warnings():
    #             warnings.simplefilter('ignore')
    #             with contextlib.redirect_stderr(open(os.devnull, 'w')):
    #
    #                 val_rmse = train_validate_model(
    #                     t_test=t_val,
    #                     x_train=x_train,
    #                     u_train=u_train,
    #                     x_dot_train=x_dot_train,
    #                     x_test=x_val,
    #                     u_test=u_val,
    #                     x_ddot_train=x_ddot_train,
    #                     model_type=model_type,
    #                     order=order,
    #                     integrator_kws=integrator_kws,
    #                     **params
    #                 )[-1]
    #
    #         return val_rmse
    #
    #     # Hyperparameter tuning
    #     rstate = np.random.Generator(np.random.PCG64(seed=42))
    #     search_space = {
    #         'alpha': hp.loguniform('alpha', np.log(1e-3), np.log(1e3)),
    #         'threshold': hp.loguniform('threshold', np.log(1e-1), np.log(1e1))
    #     }
    #
    #     best_parameters = hyperopt.space_eval(
    #         search_space,
    #         fmin(
    #             fn=validation_rmse,
    #             space=search_space,
    #             algo=tpe.suggest,
    #             max_evals=100,
    #             rstate=rstate
    #         )
    #     )
    #
    #     # TODO: save outputs
    #     # Fitting
    #     model, simulation, r2, rmse = train_validate_model(
    #         t_test=t_test,
    #         x_train=x_train,
    #         u_train=u_train,
    #         x_dot_train=x_dot_train,
    #         x_test=x_test,
    #         u_test=u_test,
    #         x_ddot_train=x_ddot_train,
    #         model_type=model_type,
    #         order=order,
    #         integrator_kws=integrator_kws,
    #         **best_parameters
    #     )
    #     model.print()
    #     print(' Scores: '.center(120, '-'))
    #     print(f'-> R2: {100 * r2:.2f}%')
    #     print(f'-> RMSE: {rmse:.4f}')
    #
    #     plt.plot(t_test, simulation[:, -1], label=readable_name)

    # 03b - Training the models based on hidden state reconstruction
    for readable_name, model_type, integrator_kws in zip(
            ['Hidden SINDy SQRT', 'Hidden SINDy'],
            ['sqrt', 'naive'],
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
                        # implicit=True,
                        **params
                    )[-1]

            return val_rmse

        # Hyperparameter tuning
        rstate = np.random.Generator(np.random.PCG64(seed=42))
        search_space = {
            # 'alpha': hp.loguniform('alpha', np.log(1e-3), np.log(1e3)),
            'threshold': hp.loguniform('threshold', np.log(1e-6), np.log(1e0)),
            # 'tol': hp.loguniform('tol', np.log(1e-7), np.log(1e-4)),
            'k1': hp.lognormal('k1', np.log(16.0), 0.01),
            'k3': hp.lognormal('k3', np.log(45.0), 0.01),
            'z0': hp.uniform('z0', 0.2, 0.7),
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
        model, simulation, r2, rmse = train_validate_hidden_model(
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
        print(f'-> RMSE: {rmse:.4f}')

        plt.plot(t_test, simulation[:, -1], label=readable_name)

    # 03c - Implicit models

    plt.legend()
    plt.show()


if __name__ == '__main__':
    Fire(cascaded_tanks)
