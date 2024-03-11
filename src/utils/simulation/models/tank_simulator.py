import numpy as np

from scipy.integrate import solve_ivp

from src.utils.functions import threshold


def tank_simulator(
        t: np.ndarray,
        u: np.ndarray,
        tf: float,
        k1: float = 3.5,
        k3: float = 12,
        z0: float = 0.5,
        lb: float = 0.0,
        ub: float = 1.0,
        eps: float = 1e-6
):
    def upper_tank_model(t_, x):
        x_dot = - k1 * np.sqrt(max(x, 0.0)) + k3 * np.interp(t_, t, u.ravel())
        if x < lb + eps:
            return max(0.0, x_dot)
        elif x > ub - eps:
            return min(0.0, x_dot)
        else:
            return x_dot

    z = solve_ivp(upper_tank_model, [0.0, tf], [z0], t_eval=t)['y'].T
    z = threshold(z, lb=lb, ub=ub).reshape(-1, 1)

    return z
