from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.utils.plotting.palettes import ibm_available_colors


def hysteresis_plot(
        u: np.ndarray,
        ys: List[np.ndarray],
        labels: List[str] = None,
        figsize_x: float = 5.25,
        figsize_y: float = 2.9
):
    available_lines = ['-', '--', '-.', ':']
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    for i, (y, label) in enumerate(zip(ys, labels)):
        plt.plot(
            u, y,
            label=label,
            zorder=len(labels) - i,
            color='k' if i == 0 else ibm_available_colors[i - 1],
            linestyle=available_lines[i % len(available_lines)]
        )

    ax = plt.gca()
    ax.margins(x=0)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=len(labels))
    plt.xlabel('Input Force [N]')
    plt.ylabel('Displacement [m]')

    return fig
