import warnings
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score

from src.utils.functions.bouc_wen import \
    identity, \
    xidentity, \
    z_xabs, \
    zabs_x, \
    zabs_xabs, \
    identity_name, \
    xidentity_name, \
    z_xabs_name, \
    zabs_x_name, \
    zabs_xabs_name


def ode_model(u: np.ndarray, t: np.ndarray, x_grad: Any, z_grad: Any):
    # preparing the model for simulation
    k, c, m = x_grad.ravel()

    def mod(t_, xyz_):
        [x_, y_, z_] = xyz_
        u_ = np.interp(t_, t, u.ravel())

        zd_ = z_grad.predict(np.array([[z_]]), u=np.array([[x_]])).ravel()[0]
        xd_ = (u_ - z_ - k * y_ - c * x_) / m
        yd_ = x_

        return np.array([xd_, yd_, zd_])

    return mod


def xdot_model(include_z: bool = True, degree: int = 3, threshold: float = 0.00001, alpha: float = 10.0):
    """simple library consisting of a linear combination of inputs and state
    """
    if include_z:
        feature_names = ['x', 'y', 'z', 'u']
    else:
        feature_names = ['x', 'y', 'u']

    lib = ps.PolynomialLibrary(
        include_bias=False,
        degree=degree,
    )
    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        feature_names=feature_names,
        differentiation_method=None,
    )
    return model, feature_names


def zdot_model(z_term: bool = True, threshold: float = 1.0, alpha: float = 1000.0):
    """library consisting of transformations akin the Bouc-Wen model
    """
    feature_names = ['z', 'x']

    functions = [
        identity if z_term else xidentity,
        z_xabs,
        zabs_x,
        zabs_xabs
    ]
    function_names = [
        identity_name if z_term else xidentity_name,
        z_xabs_name,
        zabs_x_name,
        zabs_xabs_name
    ]

    lib = ps.CustomLibrary(
        library_functions=functions,
        function_names=function_names,
        include_bias=False
    )

    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha)
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=lib,
        feature_names=feature_names,
        differentiation_method=None,
    )
    return model, feature_names


def simulate_and_score_x_model(
        m: ps.SINDy,
        t: np.ndarray,
        u: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
):
    try:
        xy_sindy = m.simulate(np.array([x[0, 0], y[0, 0]]), t, u=u)
        x_sim = np.pad(xy_sindy[:, 0], (0, 1), mode='edge')
        y_sim = np.pad(xy_sindy[:, 1], (0, 1), mode='edge')
    except Exception as e:
        warnings.warn(f'SINDy naive model failed ({type(e).__name__}: {e}), returning NaNs...')
        x_sim = np.ones((t.shape[0], )) * np.nan
        y_sim = np.ones((t.shape[0], )) * np.nan

    if x_sim.shape[0] != t.shape[0]:
        x_sim = np.pad(x_sim, (0, t.shape[0] - x_sim.shape[0]), mode='constant', constant_values=np.nan)
        y_sim = np.pad(y_sim, (0, t.shape[0] - y_sim.shape[0]), mode='constant', constant_values=np.nan)

    try:
        r2 = r2_score(y.ravel(), y_sim.ravel())
        rmse = np.sqrt(np.mean((y.ravel() - y_sim.ravel()) ** 2))
    except ValueError:
        r2 = -np.inf
        rmse = np.inf

    return x_sim.reshape(-1, 1), y_sim.reshape(-1, 1), r2, rmse


def simulate_and_score_z_model(
        m1: ps.SINDy,
        m2: np.ndarray,
        t: np.ndarray,
        u: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        xd: np.ndarray,
):
    k, c, m = m2.ravel()
    z0 = u[0, 0] - k * y[0, 0] - c * x[0, 0] - m * xd[0, 0]
    full_model = ode_model(u, t, x_grad=m2, z_grad=m1)

    try:
        xyz_sim = solve_ivp(full_model, [0.0, t[-1]], [x[0, 0], y[0, 0], z0], t_eval=t, method='LSODA')['y']
        x_sim = xyz_sim[0, :]
        y_sim = xyz_sim[1, :]
        z_sim = xyz_sim[2, :]
    except Exception as e:
        warnings.warn(f'SINDy naive model failed ({type(e).__name__}: {e}), returning NaNs...')
        x_sim = np.ones((t.shape[0],)) * np.nan
        y_sim = np.ones((t.shape[0],)) * np.nan
        z_sim = np.ones((t.shape[0],)) * np.nan

    if x_sim.shape[0] != t.shape[0]:
        x_sim = np.pad(x_sim, (0, t.shape[0] - x_sim.shape[0]), mode='constant', constant_values=np.nan)
        y_sim = np.pad(y_sim, (0, t.shape[0] - y_sim.shape[0]), mode='constant', constant_values=np.nan)
        z_sim = np.pad(z_sim, (0, t.shape[0] - z_sim.shape[0]), mode='constant', constant_values=np.nan)

    try:
        r2 = r2_score(y.ravel(), y_sim.ravel())
        rmse = np.sqrt(np.mean((y.ravel() - y_sim.ravel()) ** 2))
    except ValueError:
        r2 = -np.inf
        rmse = np.inf

    return x_sim.reshape(-1, 1), y_sim.reshape(-1, 1), z_sim.reshape(-1, 1), r2, rmse


def simulate_test(
        y_test: np.ndarray,
        u_test: np.ndarray,
        x_test: np.ndarray,
        xd_test: np.ndarray,
        naive_models: List[Any],
        hidden_models: List[Any],
        dt: float = 1 / 750.0,
        verbose: bool = True
):
    t_test = np.arange(0, y_test.shape[0]) * dt
    x_trajectories = []
    y_trajectories = []
    r2_scores = []
    rmse_scores = []

    if verbose:
        print('Simulating SINDy naive models...')
    for m0 in naive_models:
        x_naive_test, y_naive_test, naive_r2_test, naive_rmse_test = simulate_and_score_x_model(
            m0, t_test, u_test, x_test, y_test
        )
        x_trajectories.append(x_naive_test)
        y_trajectories.append(y_naive_test)
        r2_scores.append(naive_r2_test)
        rmse_scores.append(naive_rmse_test)

    if verbose:
        print('Simulating SINDy hidden models...')
    for m1, m2 in hidden_models:
        x_hidden_test, y_hidden_test, _, hidden_r2_test, hidden_rmse_test = simulate_and_score_z_model(
            m1, m2, t_test, u_test, x_test, y_test, xd_test
        )
        x_trajectories.append(x_hidden_test)
        y_trajectories.append(y_hidden_test)
        r2_scores.append(hidden_r2_test)
        rmse_scores.append(hidden_rmse_test)

    return (
        t_test,
        x_trajectories,
        y_trajectories,
        r2_scores,
        rmse_scores
    )


def high_fidelity(
        t: np.ndarray,
        u: np.ndarray,
        m: float = 2.0,
        c: float = 10.0,
        k: float = 5e4,
        alpha: float = 5e4,
        beta: float = 1e3,
        gamma: float = 0.8,
        delta: float = -1.1,
        nu: float = 1.0,
):
    # Parameter             mL      cL      kL      α       β       γ       δ       ν
    # Value (in SI unit)    2       10      5e4     5e4     1e3     0.8     -1.1    1.0
    def mod(t_, xyz_):
        [x_, y_, z_] = xyz_
        u_ = np.interp(t_, t, u.ravel())

        xd_ = (u_ - z_ - c * x_ - k * y_) / m
        if nu != 1.0:
            z_nu_m1 = np.power(np.abs(z_), nu - 1)
            z_nu = np.power(np.abs(z_), nu)
            zd_ = alpha * x_ - beta * (gamma * z_nu_m1 * np.abs(x_) * z_ - delta * x_ * z_nu)
        else:
            zd_ = alpha * x_ - beta * (gamma * np.abs(x_) * z_ - delta * x_ * np.abs(z_))
        yd_ = x_

        return np.array([xd_, yd_, zd_])

    return mod
