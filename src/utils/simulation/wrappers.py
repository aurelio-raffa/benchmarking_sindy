import numpy as np


def simulation_wrapper(
        ode_fun: callable,
        t: np.ndarray,
        u: np.ndarray,
        mode: str = 'multiinput'
):
    """Wraps a function into a format that can be used for simulation with solve_ivp.
    Notice that the time span and control input are fixed and have to be provided prior to simulation.
    """

    def ode_wrapper(t_, x_):
        # t_ is a scalar, we need to use it to interpolate the control inputs
        u_ = []
        if u.ndim == 1 or u.shape[1] == 1:
            u_.append(np.interp(t_, t, u.ravel()))
        else:
            for i in range(u.shape[1]):
                u_.append(np.interp(t_, t, u[:, i].ravel()))

        # The derivative is computed by calling the wrapped function
        if mode == 'multiinput':
            # the ode_fun expects a list of inputs
            x_d = ode_fun(x_, *u_)
        else:
            # the ode_fun expects a single array input
            x_d = ode_fun(np.array([x_] + u_))

        # We return gradient wrapped in a numpy array
        return np.array([x_d])

    return ode_wrapper
