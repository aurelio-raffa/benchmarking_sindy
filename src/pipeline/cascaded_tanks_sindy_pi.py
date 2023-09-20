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
from sklearn.metrics import r2_score
from tqdm import tqdm

from __init__ import root_path
from src.utils.import_cascaded_tanks_data import import_cascaded_tanks_data
from src.utils.symbolic_conversion import prepare_for_sympy
from src.utils.timeout import TimeoutException, timeout_handler, TimeoutManager


def compute_derivatives(
        x: np.ndarray,
        dt: float,
        tvr_gamma: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the second derivative of the data x using TV regularization.
    """
    _, x_dot = jerk(x, dt, params=[tvr_gamma])
    _, x_ddot = jerk(x_dot, dt, params=[tvr_gamma])
    return x_dot, x_ddot


def prepare_data(
        df: pd.DataFrame,
        dt: float,
        tvr_gamma: float = 10.0
):
    u = df['u'].values.reshape(-1, 1)
    y = df['y'].values
    x_dot, x_ddot = compute_derivatives(y, dt=dt, tvr_gamma=tvr_gamma)
    x = np.concatenate(
        [
            x_ddot.reshape(-1, 1),
            x_dot.reshape(-1, 1),
            y.reshape(-1, 1)
        ],
        axis=1
    )

    return x, u


def prepare_for_simulation(
        ode_fun: callable,
        t: np.ndarray,
        u: np.ndarray
):
    """Wraps a function into a format that can be used for simulation with solve_ivp.
    Notice that the time span and control input are fixed and have to be provided prior to simulation.
    """
    def ode_wrapper(t_, x_):
        # The derivative and state are saved in vector x_ in this order
        [x_d, x] = x_

        # t_ is a scalar and we need to use it to interpolate the control input
        u_ = np.interp(t_, t, u.ravel())

        # The second derivative is computed by calling the ode_fun
        x_dd = ode_fun(x_d, x, u_)

        # We return the state's second and first derivatives in this order
        return np.array([x_dd, x_d])

    return ode_wrapper


# TODO: hpo
def main(
        rescale_factor: float = 10.0,
        tvr_gamma: float = 8.0,
        threshold: float = 0.1,
        thresholder: str = "l1",
        tol: float = 1e-6,
        max_iter: int = 50000,
        precision: int = 10,
        simulation_timeout_seconds: int = 10,
):
    """Computes a second-order model with SINDy-PI for the cascaded tanks data.
    """
    # 01 - Loading the data
    train_data, test_data, t, dt, tf = import_cascaded_tanks_data()

    # TODO: rescale back when saving the results
    # 01b - Rescaling the data (for numerical stability)
    train_data /= rescale_factor
    test_data /= rescale_factor

    # 02 - Computing the derivatives (using TV regularization) and preparing the dataset
    x_train, u_train = prepare_data(train_data, dt=dt, tvr_gamma=tvr_gamma)
    x_test, u_test = prepare_data(test_data, dt=dt, tvr_gamma=tvr_gamma)

    # 03 - Definition of the library of functions to be applied to the data
    #   The library is created so as to contain only the candidate terms that we know must enter in the 2nd order model.
    #   In particular, the square root is only applied to the state, x, but not to its derivatives.
    #   Also, we do not include the square of x_ddot as we want to solve symbolically for x_ddot without ambiguity.
    #   For numerical reasons, we ensure the root can be computed by thresholding to zero.
    functions = [
        lambda x_: x_,                                                      # identity for all terms
        lambda x_, y: x_ * y,                                               # pairwise product of all terms
        lambda _, x_, __, ___: x_ ** 2,                                     # squares for all terms except x_ddot
        lambda _, __, x_, ___: x_ ** 2,
        lambda _, __, ___, x_: x_ ** 2,
        lambda _, x_, y, __: x_ * np.sqrt(np.where(y >= 0, y, 0)),          # sqrt(x) * x_dot
        lambda _, __, x_, y: np.sqrt(np.where(x_ >= 0, x_, 0)) * y,         # sqrt(x) * u
        lambda x_, y, z, _: x_ * y * np.sqrt(np.where(z >= 0, z, 0)),       # sqrt(x) * x_dot * x_ddot
    ]
    # The features and function names are very important as they allow to translate the obtained models into sympy
    # objects, which allow to solve the equations symbolically for x_ddot and obtain the models.
    feature_names = ['xdd', 'xd', 'x', 'u']
    function_names = [
        lambda x_: x_,
        lambda x_, y: x_ + ' * ' + y,
        lambda _, x_, __, ___: x_ + "**2",
        lambda _, __, x_, ___: x_ + "**2",
        lambda _, __, ___, x_: x_ + "**2",
        lambda _, x_, y, __: x_ + ' * ' + "sqrt(max(" + y + ", 0))",
        lambda _, __, x_, y: "sqrt(max(" + x_ + ", 0))" + ' * ' + y,
        lambda x_, y, z, _: x_ + ' * ' + y + ' * ' + "sqrt(max(" + z + ", 0))",
    ]
    # In the pysindy package, implicit libraries can be defined via the PDELibrary.
    # Notice that the derivative order is set to 0 since we have manually computed the derivatives.
    # Also, we do not want to allow for a bias term as we have derived the model through differentiation.
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
    model.fit(np.concatenate([x_train, u_train], axis=1))

    # 05 - Converting the implicit models to 2nd order explicit models via symbolic computation
    #   We are trying to solve for x_ddot as we assume we can control a second-order model
    model_equations = model.equations(precision=precision)
    model_features = list(np.copy(model.get_feature_names()))
    symbolic_expressions = []
    transformed_equations = []
    for i, (lhs, rhs) in enumerate(zip(model_features, model_equations)):
        fixed_rhs = prepare_for_sympy(rhs)
        print(f'original equation: {lhs} =\n\t= {fixed_rhs}')
        try:
            [symbolic_expression] = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(fixed_rhs)), sp.symbols('xdd'))
            symbolic_expressions.append(symbolic_expression)
            transformed_equations.append(f'xdd = {symbolic_expression}')
            print(f'...solved\n')
        except Exception as e:
            warnings.warn(f'Failed to solve equation {i} ({type(e).__name__}: {str(e)})!')
            continue

    # 06 - Model selection
    #   We simulate the models obtained by the previous stage on the training trajectory and select the best one in
    #   terms of RMSE.
    training_initial_conitions = x_train[0, 1:].copy()
    training_y_true = x_train[:, 2].copy()
    ode_functions = []
    training_scores = []
    training_simulations = []
    for expression in tqdm(symbolic_expressions, desc='Running simulations (training)'):
        # We first have to convert the symbolic expression into a function that can be used for computation
        ode_fun = sp.lambdify([sp.symbols('xd'), sp.symbols('x'), sp.symbols('u')], expression)
        ode_functions.append(ode_fun)

        # We prepare to run the simulation
        training_ode = prepare_for_simulation(ode_fun, t=t, u=u_train)

        # We run the simulation with a timeout manager so that unreasonably long simulations will be interrupted
        with TimeoutManager(seconds=simulation_timeout_seconds):
            try:
                y_out = solve_ivp(training_ode, [0.0, tf], training_initial_conitions, t_eval=t)['y'].T
                if y_out.shape[0] < t.shape[0]:
                    warnings.warn('Simulation did not complete, padding with last value!')
                    y_out = np.pad(y_out, ((0, t.shape[0] - y_out.shape[0]), (0, 0)), mode='edge')
                y_sim = y_out[:, 1]
                rmse = np.sqrt(np.mean((y_sim - training_y_true) ** 2))
            except TimeoutException:
                warnings.warn('Simulation timed out, returning zeros!')
                y_sim = np.nan * np.ones((t.shape[0], ))
                rmse = np.inf

        training_simulations.append(y_sim)
        training_scores.append(rmse)

    # TODO: make this prettier
    # Optional: plotting the simulated training trajectories
    plt.figure()
    plt.plot(t, training_y_true, label='true', color='black', linewidth=2)
    for i, simulation in enumerate(training_simulations):
        plt.plot(t, simulation, label=f'sim (model {i})', alpha=0.5)

    plt.ylim([-0.5, 1.5])
    plt.legend()
    plt.show()

    # 07 - Simulation on the test trajectory
    selected_model = np.argmin(training_scores)
    print(f'Best model ({training_scores[selected_model]:.3f} training RMSE): {transformed_equations[selected_model]}')

    test_initial_conitions = x_test[0, 1:].copy()
    test_y_true = x_test[:, 2].copy()
    test_ode = prepare_for_simulation(ode_functions[selected_model], t=t, u=u_test)

    y_out = solve_ivp(test_ode, [0.0, tf], test_initial_conitions, t_eval=t)['y'].T
    y_sim = y_out[:, 1]
    test_rmse = rescale_factor * np.sqrt(np.mean((y_sim - test_y_true) ** 2))
    test_r2 = r2_score(test_y_true, y_sim)

    print(f'Test RMSE: {test_rmse:.3f} | Test R2: {100 * test_r2:.3f}%')

    benchmark_data = pd.read_csv(os.path.join(root_path, 'data/Benchmarks/cascaded_tanks.csv'))

    plt.figure()
    plt.plot(t, rescale_factor * test_y_true, label='true', color='black', linewidth=2)
    plt.plot(t, rescale_factor * y_sim, label=f'simulation')
    plt.plot(t, benchmark_data['ARX'], label='ARX')
    plt.plot(t, benchmark_data['SINDy_prior'], label='SINDy_prior')
    plt.plot(t, benchmark_data['SINDy_naif'], label='SINDy_naif')
    plt.ylim([-rescale_factor * 0.2, rescale_factor * 1.2])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    Fire(main)
