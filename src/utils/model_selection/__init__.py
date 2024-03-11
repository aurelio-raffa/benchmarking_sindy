import logging
import warnings

import numpy
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.utils.etl import prepare_data
from src.utils.simulation import pad_simulation
from src.utils.simulation.models import get_model

silence_loggers = [
    "hyperopt.tpe",
    "hyperopt.fmin",
    "hyperopt.pyll.base",
]
for logger in silence_loggers:
    logging.getLogger(logger).setLevel(logging.ERROR)


def train_valid_data(
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        dt: float = 1 / 750.0,
        tvr_gamma: float = 1e-8,
        derivation_order: int = 2,
) -> tuple:
    tdata = prepare_data(training_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=derivation_order)
    vdata = prepare_data(validation_data, dt=dt, tvr_gamma=tvr_gamma, derivation_order=derivation_order)
    t_t = np.arange(0, training_data.shape[0]) * dt
    t_v = np.arange(0, validation_data.shape[0]) * dt

    return t_t, *tdata, t_v, *vdata


def score_simulation(x_true, x_sim):
    try:
        r2 = r2_score(x_true.ravel(), x_sim.ravel())
        rmse = np.sqrt(np.mean((x_true.ravel() - x_sim.ravel()) ** 2))
    except ValueError:
        r2 = -np.inf
        rmse = np.inf

    return r2, rmse


def train_validate_model(
        t_test: np.ndarray,
        x_train: np.ndarray,
        u_train: np.ndarray,
        x_dot_train: np.ndarray,
        x_test: np.ndarray,
        u_test: np.ndarray,
        model_type: str = 'naive',
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
    r2, rmse = score_simulation(x_test, simulation[:, -1])

    return model, simulation, r2, rmse
