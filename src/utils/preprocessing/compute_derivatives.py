import numpy as np

from typing import Tuple, List

from pynumdiff.total_variation_regularization import jerk


def compute_derivatives(
        x: np.ndarray,
        dt: float,
        tvr_gamma: float = 10.0,
        order: int = 2
) -> List[np.ndarray]:
    """Computes the derivatives of input data x using TV regularization.
    """
    derivatives = []
    y = x.copy()
    for _ in range(order):
        _, y = jerk(y, dt, params=[tvr_gamma])
        derivatives.append(y)
    return derivatives
