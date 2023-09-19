"""Code based on https://pysindy.readthedocs.io/en/latest/examples/9_sindypi_with_sympy/example.html#Find-complex-PDE-with-SINDy-PI-with-PDE-functionality
"""
import warnings
import re
import signal

import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import sympy as sp

from scipy.integrate import solve_ivp
from pynumdiff.total_variation_regularization import jerk


from __init__ import *


if __name__ == '__main__':
    # Load the data
    data = pd.read_csv(os.path.join(root_path, 'data/CascadedTanksFiles/dataBenchmark.csv'))
    train_data = data.loc[:, ['uEst', 'yEst']].copy().rename(columns={'uEst': 'u', 'yEst': 'y'})
    test_data = data.loc[:, ['uVal', 'yVal']].copy().rename(columns={'uVal': 'u', 'yVal': 'y'})
    t = np.linspace(0.0, 4.0, num=len(train_data))
    dt = 4.0 / len(train_data)
    print(data.describe())

    x = train_data['y'].values / 10
    x_hat, x_dot = jerk(x, dt, params=[10.0])
    x_dot_hat, x_dot_dot = jerk(x_dot, dt, params=[10.0])
    u_train = train_data['u'].values.reshape(-1, 1) / 10
    #
    # df_new = pd.DataFrame(
    #     {'x': x, 'x_dot': x_dot, 'x_dot_dot': x_dot_dot, 'u': u_train.ravel(), 't': t, 'x_hat': x_hat, 'x_dot_hat': x_dot_hat}
    # )
    # print(df_new)
    # df_new = df_new.loc[np.logical_and(df_new['x'] > 0.0, df_new['x'] < 1.0)]
    # [x, x_dot, x_dot_dot, u_train, t, x_hat, x_dot_hat] = [
    #     df_new['x'].values, df_new['x_dot'].values, df_new['x_dot_dot'].values, df_new['u'].values, df_new['t'].values,
    #     df_new['x_hat'].values, df_new['x_dot_hat'].values
    # ]

    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ax1.plot(t, x_dot_dot, label='x_dot_dot')
    ax2.plot(t, x_dot_hat, label='x_dot_hat')
    ax2.plot(t, x_dot, label='x_dot')
    ax3.plot(t, x_hat, label='x_hat')
    ax3.plot(t, x, label='x')
    ax4.plot(t, u_train, label='u')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.show()

    # Functions to be applied to the data x_dot
    x_dot_functions = [lambda x: x]

    # Functions to be applied to the data x
    functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda _, x, __, ___: x ** 2,
        lambda _, __, x, ___: x ** 2,
        lambda _, __, ___, x: x ** 2,
        # lambda x: np.sqrt(np.abs(x)),
        # lambda x: x * np.sqrt(np.abs(x)),
        lambda _, x, y, __: x * np.sqrt(np.where(y >= 0, y, 0)),
        lambda _, __, x, y: np.sqrt(np.where(x >= 0, x, 0)) * y,
        # lambda x, y, z: np.sqrt(np.abs(x)) * y * z,
        lambda x, y, z, _: x * y * np.sqrt(np.where(z >= 0, z, 0)),
        # lambda x, y, z: x * y * np.sqrt(z)
    ]
    function_names = [
        lambda x: x,
        lambda x, y: x + ' * ' + y,
        lambda _, x, __, ___: x + "**2",
        lambda _, __, x, ___: x + "**2",
        lambda _, __, ___, x: x + "**2",
        # lambda x: "sqrt(" + x + ")",
        # lambda x: x + "sqrt(" + x + ")",
        lambda _, x, y, __: x + ' * ' + "sqrt(max(" + y + ", 0))",
        lambda _, __, x, y: "sqrt(max(" + x + ", 0))" + ' * ' + y,
        # lambda x, y, z: "sqrt(" + x + ")" + y + z,
        lambda x, y, z, _: x + ' * ' + y + ' * ' + "sqrt(max(" + z + ", 0))",
        # lambda x, y, z: x + y + "sqrt(" + z + ")",
    ]
    # function_names = [
    #     lambda x: x,
    #     lambda x, y: x + y,
    #     lambda _, x, __, ___: x + "**2",
    #     lambda _, __, x, ___: x + "**2",
    #     lambda _, __, ___, x: x + "**2",
    #     # lambda x: "sqrt(" + x + ")",
    #     # lambda x: x + "sqrt(" + x + ")",
    #     lambda _, x, y, __: x + "sqrt(" + y + ")",
    #     lambda _, __, x, y: "sqrt(" + x + ")" + y,
    #     # lambda x, y, z: "sqrt(" + x + ")" + y + z,
    #     lambda x, y, z, _: x + y + "sqrt(" + z + ")",
    #     # lambda x, y, z: x + y + "sqrt(" + z + ")",
    # ]

    lib = ps.PDELibrary(
        library_functions=functions,
        derivative_order=0,
        function_names=function_names,
        include_bias=False,
        implicit_terms=True,
        temporal_grid=t
    )

    # lib = sindy_library = ps.SINDyPILibrary(
    #     library_functions=functions,
    #     x_dot_library_functions=[lambda x: x],
    #     t=t,
    #     function_names=[lambda x: x] + function_names,
    #     include_bias=True,
    # )

    sindy_opt = ps.SINDyPI(
        threshold=1e-1,
        tol=1e-6,
        thresholder="l1",
        max_iter=20000,
    )

    features = ['xdd', 'xd', 'x', 'u']
    model = ps.SINDy(
        optimizer=sindy_opt,
        feature_library=lib,
        feature_names=features,
        differentiation_method=ps.FiniteDifference(drop_endpoints=True),
    )

    x_train = np.concatenate(
        [
            x_dot_dot.reshape(-1, 1),
            x_dot.reshape(-1, 1),
            x.reshape(-1, 1)
        ],
        axis=1
    )
    # print(lib.fit(np.concatenate([x_train, u_train], axis=1)).get_feature_names())
    # exit( 1)

    model.fit(x_train, t=t, u=u_train)
    # model.print()


    # # Need to put multiplication between terms for sympy
    # print(features)
    # model_features = list(np.copy(model.get_feature_names()))
    # print(model_features)
    # nfeatures = len(features)
    # coefs = model.coefficients()
    # sym_features = np.array([sp.symbols(feature) for feature in features])
    # sym_theta = np.array([sp.symbols(feature) for feature in model_features])
    #
    # print(sym_theta[0])
    # print(sym_theta)
    # print(np.around(coefs[0], 10))
    # print(sym_theta @ np.around(coefs[0], 10))

    def make_sympifiable(equation: str):
        equation = equation.replace('+ -', '- ')
        return re.sub(r'(\d+\.\d+)\s([a-zA-Z]+)', r'\1 * \2', equation)


    model_equations = model.equations(precision=5)
    model_features = list(np.copy(model.get_feature_names()))
    odes = []
    simulations = []
    for lhs, rhs in zip(model_features, model_equations):
        fixed_rhs = make_sympifiable(rhs)
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

        class TimeoutException(Exception): pass

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