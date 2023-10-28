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


def prepare_for_simulation(
        ode_fun: callable,
        t: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
):
    """Wraps a function into a format that can be used for simulation with solve_ivp.
    Notice that the time span and control input are fixed and have to be provided prior to simulation.
    """

    def ode_wrapper(t_, x_):
        # t_ is a scalar and we need to use it to interpolate the control input
        z_ = np.interp(t_, t, z.ravel())
        u_ = np.interp(t_, t, u.ravel())

        # The second derivative is computed by calling the ode_fun
        x_d = ode_fun(x_, z_, u_)

        # We return the state's second and first derivatives in this order
        return np.array([x_d])

    return ode_wrapper


def simulate_and_select(
        model,
        t: np.ndarray,
        tf: float,
        z_train: np.ndarray,
        u_train: np.ndarray,
        training_feats: np.ndarray,
        precision: int = 10,
        simulation_timeout_seconds: int = 10
):
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
            [symbolic_expression] = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(fixed_rhs)), sp.symbols('xd'))
            symbolic_expressions.append(symbolic_expression)
            transformed_equations.append(f'xd = {symbolic_expression}')
            print(f'...solved\n')
        except Exception as e:
            warnings.warn(f'Failed to solve equation {i} ({type(e).__name__}: {str(e)})!')
            continue

    # 06 - Model selection
    #   We simulate the models obtained by the previous stage on the training trajectory and select the best one in
    #   terms of RMSE.
    training_initial_conitions = training_feats[0, 1:2].copy()
    training_y_true = training_feats[:, 1].copy()
    ode_functions = []
    training_scores = []
    training_simulations = []
    for expression in tqdm(symbolic_expressions, desc='Running simulations (training)'):
        # We first have to convert the symbolic expression into a function that can be used for computation
        ode_fun = sp.lambdify([sp.symbols('x'), sp.symbols('z'), sp.symbols('u')], expression)
        ode_functions.append(ode_fun)

        # We prepare to run the simulation
        training_ode = prepare_for_simulation(ode_fun, t=t, z=z_train, u=u_train)

        # We run the simulation with a timeout manager so that unreasonably long simulations will be interrupted
        with TimeoutManager(seconds=simulation_timeout_seconds):
            try:
                y_out = solve_ivp(training_ode, [0.0, tf], training_initial_conitions, t_eval=t)['y'].T
                if y_out.shape[0] < t.shape[0]:
                    warnings.warn('Simulation did not complete, padding with last value!')
                    y_out = np.pad(y_out, ((0, t.shape[0] - y_out.shape[0]), (0, 0)), mode='edge')
                y_sim = y_out[:, 0]
                mse = np.mean((y_sim - training_y_true) ** 2)
            except TimeoutException:
                warnings.warn('Simulation timed out, returning zeros!')
                y_sim = np.nan * np.ones((t.shape[0],))
                mse = np.inf

        training_simulations.append(y_sim)
        training_scores.append(mse)

    # We select the best model based on the RMSE
    best_model_idx = np.argmin(training_scores)
    best_score = training_scores[best_model_idx]
    best_model = transformed_equations[best_model_idx]
    best_sim = training_simulations[best_model_idx]

    return best_score, best_sim, best_model


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
    feature_names = ['xd', 'x', 'z', 'u']
    functions = [
        lambda x: x,  # identity for all terms
        # lambda x: x ** 2,
        lambda _, x, __, ___: np.sqrt(double_thresh(x)),
        lambda _, __, z, ___: np.sqrt(double_thresh(z)),
        lambda _, __, ___, u: np.sqrt(double_thresh(u)),
        lambda xd, x, __, ___: xd / np.sqrt(double_thresh(x)),  # x_dot / sqrt(x)
        lambda _, __, z, u: u / np.sqrt(double_thresh(z)),
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
        lambda xd, x, __, ___: f'{xd} / sqrt(min(max({x}, 0), {max_val}))',  # x_dot / sqrt(x)
        lambda _, __, z, u: f'{u} / sqrt(min(max({z}, 0), {max_val}))',  # u / sqrt(z)
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
    [k1_opt, k3_opt, z0_opt] = [19, 41.75, 0.5]

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
    print(f'final score: {score}')
    print(f'r2: {r2_score(x_train.ravel(), simulation.ravel())}')

    plt.plot(t, x_train, label='true')
    plt.plot(t, z_train, label='hidden')
    plt.plot(t, simulation, label='sim')
    plt.legend()
    plt.show()

    exit()

    # plt.plot(t, x_train[:, 0], label='true')
    # plt.plot(t, z, label='sim')
    # plt.legend()
    # plt.show()

    lr = LinearRegression(fit_intercept=True)

    # feats = np.concatenate([np.sqrt(x_train), np.sqrt(z)], axis=1)
    # feats = np.concatenate([x_train, np.sqrt(x_train), np.sqrt(u_train)], axis=1)
    z_shift = z_train.copy()
    z_shift[1:, :] -= z_train[:-1, :]
    feats = np.concatenate([x_train, np.sqrt(x_train), z_shift / dt], axis=1)
    # feats = np.concatenate([np.sqrt(x_train), (np.gradient(z.ravel()) / dt).reshape(-1, 1)], axis=1)
    # feats = np.concatenate([np.sqrt(x_train), np.sqrt(z), u_train], axis=1)

    s1 = StandardScaler()
    s2 = StandardScaler()

    # feats = s1.fit_transform(feats)
    # x_dot_train = s2.fit_transform(x_dot_train)
    # feats = np.concatenate([jerk(np.sqrt(x_train).ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1),
    #                         jerk(np.sqrt(z).ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1)], axis=1)

    lr.fit(feats[:-1, :], x_train.ravel()[1:])
    # plt.plot(t, x_dot_train, label='true')
    # # plt.plot(t, x_train)
    # # plt.plot(t, z)
    # plt.plot(t, lr.predict(feats), label='preds')
    # plt.plot(t, feats[:, 0])
    # plt.plot(t, feats[:, 1])
    # plt.legend()
    # plt.show()

    # preds = -jerk(lr.predict(feats), dt, params=[tvr_gamma])[1]
    # print(r2_score(x_dot_train.ravel(), preds.ravel()))
    print(lr.intercept_)
    print(lr.coef_)

    sim = np.zeros_like(t)
    bias = -0.050734280769437756
    weights = np.array([0.05321448, 0.06418771])
    sim[0] = x_train[0, 0]
    for i in range(1, sim.shape[0]):
        sim[i] = bias + 0.96082371 * sim[i - 1] + np.dot(weights,
                                                         np.array([np.sqrt(sim[i - 1]), np.sqrt(u_train[i - 1])]))

    plt.plot(t, sim, label='sim')
    plt.plot(t[1:], x_train[1:, :], label='true')
    # plt.plot(t, x_dot_train, label='true der')
    plt.plot(t[1:], lr.predict(feats[:-1, :]).ravel(), label='preds')
    # plt.plot(t, np.cumsum(x_dot_train.ravel()) * dt, label='der tv')
    # plt.plot(t, np.cumsum(preds.ravel()) * dt, label='preds der int')
    # plt.plot(t, -lr.predict(feats).ravel(), label='?')
    plt.legend()
    plt.show()

    plt.plot(t, x_dot_train, label='test')
    plt.plot(t, preds, label='preds')
    # plt.plot(t, -np.gradient(preds.ravel()) / dt, label='preds2')
    plt.legend()
    plt.show()

    # 04 - Fitting the model on the estimation data
    sindy_opt = ps.STLSQ(
        # threshold=threshold,
        # alpha=1.0,
        max_iter=max_iter,
    )
    model = ps.SINDy(
        optimizer=sindy_opt,
        feature_library=lib,
        feature_names=feature_names
    )
    model.fit(x_train, u=z_train, t=t, x_dot=x_dot_train)
    model.print()

    x_sim = model.simulate(x_train[0, :], t, u=z_train)
    x_sim = jerk(np.sqrt(x_sim).ravel(), dt, params=[tvr_gamma])[1].reshape(-1, 1)
    plt.plot(t, x_train, label='true')
    plt.plot(t[:-1], x_sim, label='sim')
    plt.legend()
    plt.show()

    #
    # #   The library is created so as to contain only the candidate terms that we know must enter in the 2nd order model.
    # #   In particular, the square root is only applied to the state, x, but not to its derivatives.
    # #   Also, we do not include the square of x_ddot as we want to solve symbolically for x_ddot without ambiguity.
    # #   For numerical reasons, we ensure the root can be computed by thresholding to zero.
    #
    # functions = [
    #     lambda x_, _, __, ___: x_,  # identity for all terms
    #     lambda _, x_, __, ___: x_,
    #     lambda _, __, x_, ___: double_thresh(x_) ** 2,
    #     lambda _, __, ___, x_: x_,
    #     lambda x_, y, __, ___: x_ * y,  # pairwise product of all terms
    #     lambda x_, _, y, ___: x_ * double_thresh(y),
    #     lambda x_, _, __, y: x_ * y,
    #     lambda _, x_, y, __: x_ * double_thresh(y),
    #     lambda _, x_, __, y: x_ * y,
    #     lambda _, __, x_, y: double_thresh(x_) * y,
    #     lambda _, x_, __, ___: x_ ** 2,  # squares for all terms except x_ddot
    #     lambda _, __, x_, ___: double_thresh(x_) ** 2,
    #     lambda _, __, ___, x_: x_ ** 2,
    #     lambda _, x_, y, __: x_ * np.sqrt(double_thresh(y)),  # sqrt(x) * x_dot
    #     lambda _, __, x_, y: np.sqrt(double_thresh(x_)) * y,  # sqrt(x) * u
    #     lambda x_, y, z, _: x_ * y * np.sqrt(double_thresh(z)),  # sqrt(x) * x_dot * x_ddot
    # ]
    # # The features and function names are very important as they allow to translate the obtained models into sympy
    # # objects, which allow to solve the equations symbolically for x_ddot and obtain the models.
    # feature_names = ['xdd', 'xd', 'x', 'u']
    # function_names = [
    #     lambda x_, _, __, ___: x_,  # identity for all terms
    #     lambda _, x_, __, ___: x_,
    #     lambda _, __, x_, ___: f'min(max({x_}, 0), {max_val})',
    #     lambda _, __, ___, x_: x_,
    #     lambda x_, y, __, ___: f'{x_} * {y}',  # pairwise product of all terms
    #     lambda x_, _, y, ___: f'{x_} * min(max({y}, 0), {max_val})',
    #     lambda x_, _, __, y: f'{x_} * {y}',
    #     lambda _, x_, y, __: f'{x_} * min(max({y}, 0), {max_val})',
    #     lambda _, x_, __, y: f'{x_} * {y}',
    #     lambda _, __, x_, y: f'min(max({x_}, 0), {max_val}) * {y}',
    #     lambda _, x_, __, ___: x_ + "**2",
    #     lambda _, __, x_, ___: f"min(max({x_}, 0), {max_val})**2",
    #     lambda _, __, ___, x_: x_ + "**2",
    #     lambda _, x_, y, __: x_ + ' * ' + f"sqrt(min(max({y}, 0), {max_val}))",
    #     lambda _, __, x_, y: f"sqrt(min(max({x_}, 0), {max_val}))" + ' * ' + y,
    #     lambda x_, y, z, _: x_ + ' * ' + y + ' * ' + f"sqrt(min(max({z}, 0), {max_val}))",
    # ]
    # # In the pysindy package, implicit libraries can be defined via the PDELibrary.
    # # Notice that the derivative order is set to 0 since we have manually computed the derivatives.
    # # Also, we do not want to allow for a bias term as we have derived the model through differentiation.
    # lib = ps.PDELibrary(
    #     library_functions=functions,
    #     derivative_order=0,
    #     function_names=function_names,
    #     include_bias=False,
    #     implicit_terms=True,
    #     temporal_grid=t
    # )
    #
    # # 04 - Fitting the model on the estimation data
    # #   The SINDyPI solver will attempt to fit one implicit model for each candidate term.
    # #   Beware that this could quickly get out of hand for large candidate libraries.
    # sindy_opt = ps.SINDyPI(
    #     threshold=threshold,
    #     tol=tol,
    #     thresholder=thresholder,
    #     max_iter=max_iter,
    # )
    # model = ps.SINDy(
    #     optimizer=sindy_opt,
    #     feature_library=lib,
    #     feature_names=feature_names,
    #     differentiation_method=ps.FiniteDifference(drop_endpoints=True),
    # )
    # model.fit(np.concatenate([x_train, u_train], axis=1))
    #
    # # 05 - Converting the implicit models to 2nd order explicit models via symbolic computation
    # #   We are trying to solve for x_ddot as we assume we can control a second-order model
    # model_equations = model.equations(precision=precision)
    # model_features = list(np.copy(model.get_feature_names()))
    # symbolic_expressions = []
    # transformed_equations = []
    # for i, (lhs, rhs) in enumerate(zip(model_features, model_equations)):
    #     fixed_rhs = prepare_for_sympy(rhs)
    #     print(f'original equation: {lhs} =\n\t= {fixed_rhs}')
    #     try:
    #         [symbolic_expression] = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(fixed_rhs)), sp.symbols('xdd'))
    #         symbolic_expressions.append(symbolic_expression)
    #         transformed_equations.append(f'xdd = {symbolic_expression}')
    #         print(f'...solved\n')
    #     except Exception as e:
    #         warnings.warn(f'Failed to solve equation {i} ({type(e).__name__}: {str(e)})!')
    #         continue
    #
    # # 06 - Model selection
    # #   We simulate the models obtained by the previous stage on the training trajectory and select the best one in
    # #   terms of RMSE.
    # training_initial_conitions = x_train[0, 1:].copy()
    # training_y_true = x_train[:, 2].copy()
    # ode_functions = []
    # training_scores = []
    # training_simulations = []
    # for expression in tqdm(symbolic_expressions, desc='Running simulations (training)'):
    #     # We first have to convert the symbolic expression into a function that can be used for computation
    #     ode_fun = sp.lambdify([sp.symbols('xd'), sp.symbols('x'), sp.symbols('u')], expression)
    #     ode_functions.append(ode_fun)
    #
    #     # We prepare to run the simulation
    #     training_ode = prepare_for_simulation(ode_fun, t=t, u=u_train)
    #
    #     # We run the simulation with a timeout manager so that unreasonably long simulations will be interrupted
    #     with TimeoutManager(seconds=simulation_timeout_seconds):
    #         try:
    #             y_out = solve_ivp(training_ode, [0.0, tf], training_initial_conitions, t_eval=t)['y'].T
    #             if y_out.shape[0] < t.shape[0]:
    #                 warnings.warn('Simulation did not complete, padding with last value!')
    #                 y_out = np.pad(y_out, ((0, t.shape[0] - y_out.shape[0]), (0, 0)), mode='edge')
    #             y_sim = y_out[:, 1]
    #             rmse = np.sqrt(np.mean((y_sim - training_y_true) ** 2))
    #         except TimeoutException:
    #             warnings.warn('Simulation timed out, returning zeros!')
    #             y_sim = np.nan * np.ones((t.shape[0],))
    #             rmse = np.inf
    #
    #     training_simulations.append(y_sim)
    #     training_scores.append(rmse)
    #
    # # TODO: make this prettier
    # # Optional: plotting the simulated training trajectories
    # plt.figure()
    # plt.plot(t, training_y_true, label='true', color='black', linewidth=2)
    # for i, simulation in enumerate(training_simulations):
    #     plt.plot(t, simulation, label=f'sim (model {i})', alpha=0.5)
    #
    # plt.ylim([-0.5, 1.5])
    # plt.legend()
    # plt.show()
    #
    # # 07 - Simulation on the test trajectory
    # selected_model = np.argmin(training_scores)
    # print(f'Best model ({training_scores[selected_model]:.3f} training RMSE): {transformed_equations[selected_model]}')
    #
    # test_initial_conitions = x_test[0, 1:].copy()
    # test_y_true = x_test[:, 2].copy()
    # test_ode = prepare_for_simulation(ode_functions[selected_model], t=t, u=u_test)
    #
    # y_out = solve_ivp(test_ode, [0.0, tf], test_initial_conitions, t_eval=t)['y'].T
    # y_sim = y_out[:, 1]
    # test_rmse = rescale_factor * np.sqrt(np.mean((y_sim - test_y_true) ** 2))
    # test_r2 = r2_score(test_y_true, y_sim)
    #
    # print(f'Test RMSE: {test_rmse:.3f} | Test R2: {100 * test_r2:.3f}%')
    #
    # benchmark_data = pd.read_csv(os.path.join(root_path, 'data/Benchmarks/cascaded_tanks.csv'))
    #
    # plt.figure()
    # plt.plot(t, rescale_factor * test_y_true, label='true', color='black', linewidth=2)
    # plt.plot(t, rescale_factor * y_sim, label=f'simulation')
    # plt.plot(t, benchmark_data['ARX'], label='ARX')
    # plt.plot(t, benchmark_data['SINDy_prior'], label='SINDy_prior')
    # plt.plot(t, benchmark_data['SINDy_naif'], label='SINDy_naif')
    # plt.ylim([-rescale_factor * 0.2, rescale_factor * 1.2])
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    Fire(main)
