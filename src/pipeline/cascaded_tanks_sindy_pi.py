"""Code based on https://pysindy.readthedocs.io/en/latest/examples/9_sindypi_with_sympy/example.html#Find-complex-PDE-with-SINDy-PI-with-PDE-functionality
"""
import warnings
import signal

import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import sympy as sp

from typing import Tuple
from fire import Fire
from scipy.integrate import solve_ivp
from pynumdiff.total_variation_regularization import jerk

from src.utils.import_cascaded_tanks_data import import_cascaded_tanks_data
from src.utils.symbolic_conversion import prepare_for_sympy


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


def main(
        rescale_factor: float = 10.0,
        tvr_gamma: float = 10.0,
        threshold: float = 1e-1,
        thresholder: str = "l1",
        tol: float = 1e-6,
        max_iter: int = 20000,
):
    """Computes a second-order model with SINDy-PI for the cascaded tanks data.
    """
    # 01 - Loading the data
    train_data, test_data, t, dt = import_cascaded_tanks_data()

    # TODO: rescale back when saving the results
    # 01b - Rescaling the data (for numerical stability)
    train_data /= rescale_factor
    test_data /= rescale_factor

    # 02 - Computing the derivatives (using TV regularization) and preparing the dataset
    u_train = train_data['u'].values.reshape(-1, 1)
    x = train_data['y'].values
    x_dot, x_ddot = compute_derivatives(x, dt=dt, tvr_gamma=tvr_gamma)
    x_train = np.concatenate(
        [
            x_ddot.reshape(-1, 1),
            x_dot.reshape(-1, 1),
            x.reshape(-1, 1)
        ],
        axis=1
    )

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
    model.fit(x_train, t=t, u=u_train)

    # 05 - Converting the implicit models to 2nd order explicit models via symbolic computation
    model_equations = model.equations(precision=5)
    model_features = list(np.copy(model.get_feature_names()))
    odes = []
    simulations = []
    for lhs, rhs in zip(model_features, model_equations):
        fixed_rhs = prepare_for_sympy(rhs)
        print(f'original equation:\n\t{lhs} = {fixed_rhs}')
        [symbolic_expression] = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(fixed_rhs)), sp.symbols('xdd'))
        print(f'solved as:\n\t{symbolic_expression}')

        # TODO: construct a function you can feed to odeint for each of this expressions
        ode = sp.lambdify([sp.symbols('xd'), sp.symbols('x'), sp.symbols('u')], symbolic_expression)

        def ode_wrapper(t_, x_, enforce_bounds: bool = False):
            eps = 1e-7
            u_ = np.interp(t_, t, u_train.ravel())
            x_dd = ode(x_[0], x_[1], u_)

            # TODO: fix this
            x_d = x_[0]
            if enforce_bounds:
                if x_[1] < 0.0 + eps:
                    print(f'hit lower boundary @ t = {t_:.3f}: x = {x_[1]:.3f}, v = {x_[0]:.3f}, a = {x_dd}')
                    # hardcoded boundaries
                    x_d = 0.0
                    if x_dd < 0.0 + eps:
                        x_dd = 0.0
                elif x_[1] > 1.0 - eps:
                    print(f'hit upper boundary @ t = {t_:.3f}: x = {x_[1]:.3f}, v = {x_[0]:.3f}, a = {x_dd}')
                    # hardcoded boundaries
                    x_d = 0.0
                    if x_dd > 0.0 + eps:
                        x_dd = 0.0
            return np.array([x_dd, x_d])

        class TimeoutException(Exception):
            pass

        def handler(signum, frame):
            warnings.warn("Timeout reached")
            raise TimeoutException("Timeout reached")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(20)

        try:
            y_out = solve_ivp(ode_wrapper, [0.0, 4.0], np.array([x_dot[0], x[0]]), t_eval=t).y.T
        except TimeoutException:
            print('Simulation timed out')
            y_out = np.zeros((len(t), 2))

        print(y_out.shape)

        if y_out.shape[0] != len(t):
            print('padding needed')
            y_out = np.pad(y_out, ((0, len(t) - y_out.shape[0]), (0, 0)), 'edge')
            print(y_out.shape)

        simulations.append(y_out[:, 1])

    plt.figure()
    plt.plot(t, x, label='true', color='black', linewidth=2)
    for i, simulation in enumerate(simulations):
        plt.plot(t, simulation, label=f'sim (model {i})', alpha=0.5)

    plt.ylim([-0.5, 1.5])
    plt.legend()
    plt.show()

    # for i in range(nfeatures):
    #     sym_equations.append(
    #         sp.solve(
    #             sp.Eq(sym_theta[i], sym_theta @ np.around(coefs[i], 10)), sym_features[i]
    #         )
    #     )
    #     sym_equations_rounded.append(
    #         sp.solve(
    #             sp.Eq(sym_theta[i], sym_theta @ np.around(coefs[i], 2)), sym_features[i]
    #         )
    #     )
    #     print(sym_theta[i], " = ", sym_equations_rounded[i][0])


if __name__ == '__main__':
    Fire(main)
