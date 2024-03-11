import warnings

import pandas as pd

from copy import deepcopy
from pynumdiff import jerk

from src.utils.functions.bouc_wen import *
from src.utils.model_selection import train_valid_data
from src.utils.simulation.bouc_wen import xdot_model, zdot_model, simulate_and_score_x_model, simulate_and_score_z_model


def train_and_validate_x_model(
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        dt: float = 1 / 750.0,
        tvr_gamma: float = 1e-8,
        verbose: bool = True,
        threshold: float = 0.00001,
        alpha: float = 10.0,
):
    _, y_t, u_t, x_t, xd_t, t_v, y_v, u_v, x_v, xd_v = train_valid_data(
        training_data,
        validation_data,
        dt=dt,
        tvr_gamma=tvr_gamma,
        derivation_order=2
    )

    m0, _ = xdot_model(include_z=False, threshold=threshold, alpha=alpha)
    m0.fit(np.concatenate([x_t, y_t], axis=1), u=u_t, x_dot=np.concatenate([xd_t, x_t], axis=1))

    try:
        x_sindy, y_sindy, sindy_r2, sindy_rmse = simulate_and_score_x_model(
            m0, t_v, u_v, x_v, y_v
        )
        if verbose:
            print('=' * 120)
            print(f'SINDy naive model:')
            m0.print()
            print('-' * 120)
            print(f'\tR2: {100 * sindy_r2:.3f}%, RMSE: {sindy_rmse:.3e}')

        return m0, x_sindy, y_sindy, sindy_rmse, sindy_r2
    except Exception:
        warnings.warn('SINDy naive model failed, terminating prematurely...')
        return None, None, None, np.inf, -np.inf


def train_and_validate_z_model(
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        dt: float = 1 / 750.0,
        tvr_gamma: float = 1e-8,
        k: float = 5e4,
        c: float = 10.0,
        m: float = 2.0,
        tvr_gamma_z: float = 0.1,
        verbose: bool = True,
        z_threshold: float = 1.0,
        alpha: float = 1e-8,
        z_term: bool = False,
):
    _, y_t, u_t, x_t, xd_t, t_v, y_v, u_v, x_v, xd_v = train_valid_data(
        training_data,
        validation_data,
        dt=dt,
        tvr_gamma=tvr_gamma,
        derivation_order=2
    )

    # We use the first model equation to recover the hidden state z from the initial guess
    # xd = (u - k * y - c * x - z) / m
    z_est = u_t - k * y_t - c * x_t - m * xd_t
    z_ = z_est.copy()
    m1, m2, m2_coef = None, None, None
    x_full, y_full, z_full = None, None, None
    z_mod_rmse, z_mod_r2, z = np.inf, - np.inf, None

    # First stage: we use the z estimate to fit a dynamical model based on perfect knowledge of x_dot
    try:
        z_, zd = jerk(z_.ravel(), dt, params=[tvr_gamma_z])
    except Exception as e:
        warnings.warn(f'Derivative estimation failed ({type(e).__name__}: {e}), terminating prematurely...')
        return m1, m2_coef, x_full, y_full, z_full, z_mod_rmse, z_mod_r2
    z_ = z_.reshape(-1, 1)
    zd = zd.reshape(-1, 1)

    m1_, _ = zdot_model(z_term=z_term, threshold=z_threshold, alpha=alpha)
    m1_.fit(z_est, u=x_t, x_dot=zd)
    m2_ = np.array([k, c, m])

    if verbose:
        print('SINDy hidden model:')
        m1_.print()
        print(f"(x)' = {- c / m:.3f} x + {- k / m:.3f} y + {- 1 / m:.3f} z + {1 / m:.3f} u")
        print("(y)' = 1.000 x")
        print('-' * 120)

    try:
        # Using the model for simulation - here we combine the previous two models, reducing the controls
        # to u exclusively - notice that we simulate on the validation trajectory.
        # To recover the initialization of the hidden state we use the first model equation again,
        # with derivatives up to the second order of the output y on the first validation datapoint.
        x_full_, y_full_, z_full_, z_mod_r2_, z_mod_rmse_ = simulate_and_score_z_model(
            m1_, np.array([k, c, m]), t_v, u_v, x_v, y_v, xd_v
        )

        if verbose:
            print(f'\tR2: {100 * z_mod_r2_:.3f}%, RMSE: {z_mod_rmse_:.3e}')

        # When using the process iteratively, we save the best model in terms of validation RMSE
        if z_mod_rmse is None or z_mod_rmse_ < z_mod_rmse:
            m1, m2_coef = deepcopy(m1_), deepcopy(m2_)
            x_full, y_full, z_full = x_full_.copy(), y_full_.copy(), z_full_.copy()
            z_mod_rmse, z_mod_r2, z = z_mod_rmse_, z_mod_r2_, z_.copy()

    except Exception:
        warnings.warn(f'Simulation failed, terminating experiment...')

    return m1, m2_coef, x_full, y_full, z_full, z_mod_rmse, z_mod_r2
