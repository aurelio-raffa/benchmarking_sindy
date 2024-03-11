import numpy as np


def pad_simulation(simulation, t):
    if simulation.shape[0] != t.shape[0]:
        simulation = np.pad(
            simulation,
            ((0, t.shape[0] - simulation.shape[0]), (0, 0)),
            mode='constant',
            constant_values=np.nan
        )

    return simulation
