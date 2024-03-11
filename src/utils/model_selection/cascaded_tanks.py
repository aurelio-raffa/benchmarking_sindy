import warnings

import numpy as np
from sklearn.metrics import r2_score

from __init__ import *
from src.utils.model_selection import score_simulation
from src.utils.simulation import pad_simulation
from src.utils.simulation.models import get_model
from src.utils.simulation.models.tank_simulator import tank_simulator


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
        integrator_kws=None,
        **model_kwargs
):
    # -> instantiating the model
    if integrator_kws is None:
        integrator_kws = {}
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

    # step 2: fitting the model on the hidden state
    model.fit(x_train, u=z_train, x_dot=x_dot_train)

    # -> simulating
    # we recover the (hidden) initial condition from the first sample of the state derivative
    n_trials = 100
    z0_trials = np.linspace(0.0, 1.0, n_trials)
    xd_test_pred = model.predict(np.repeat(x_test[0, :], repeats=n_trials, axis=0), u=z0_trials)
    id_min = np.argmin(np.abs(xd_test_pred - x_dot_test[0, 0]))
    z0_test = z0_trials[id_min]

    # we simulate the two models in cascade
    try:
        z_sim = tank_simulator(
            t=t_test,
            u=u_test,
            tf=t_test[-1],
            k1=k1,
            k3=k3,
            z0=z0_test
        )
        simulation = model.simulate(
            x_test[0, :], t=t_test, u=z_sim, integrator_kws=integrator_kws
        )
        simulation = np.pad(simulation.ravel(), (0, 1), 'edge').reshape(-1, 1)
    except Exception as e:
        warnings.warn(f'SINDy naive model failed ({type(e).__name__}: {e}), returning NaNs...')
        simulation = np.ones((t_test.shape[0], 1)) * np.nan
        z_sim = np.ones((t_test.shape[0], 1)) * np.nan

    # -> scoring
    simulation = pad_simulation(simulation, t_test)
    r2, rmse = score_simulation(x_test, simulation[:, -1])

    return model, simulation, z_sim, r2, rmse
