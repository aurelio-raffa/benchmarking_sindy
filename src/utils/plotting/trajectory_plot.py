from typing import List, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.utils.plotting.palettes import ibm_available_colors


def trajectory_plot(
        t: np.ndarray,
        xs: List[np.ndarray],
        labels: List[str],
        ys: List[np.ndarray] = None,
        u: np.ndarray = None,
        figsize_x: float = 8,
        figsize_y: float = 2.9,
        x_subplot_label: Union[str, None] = 'State derivative',
        y_subplot_label: Union[str, None] = 'State',
        u_subplot_label: Union[str, None] = 'Control Input',
        time_scale: str = 'Time [s]',
        x_scale: str = None,
        y_scale: str = None,
        u_scale: str = None,
        legend_outside_plots: bool = True,
        scientific_notation: bool = True
):
    assert all(x.shape[0] == t.shape[0] for x in xs), 'Inconsistent number of samples!'
    assert len(labels) == len(xs), 'Different number of labels and trajectories!'
    if u is not None:
        assert t.shape[0] == u.shape[0], 'Inconsistent number of samples!'
    if ys is not None:
        assert len(xs) == len(ys), 'Different number of trajectories for x and y!'
        assert all(x.shape[0] == y.shape[0] for x, y in zip(xs, ys)), 'Inconsistent number of samples!'
    if ys is not None:
        assert len(xs) == len(ys), 'Different number of trajectories for x and y!'
        assert all(x.shape[0] == t.shape[0] == y.shape[0] for x, y in zip(xs, ys)), 'Inconsistent number of samples!'

    available_lines = ['-', '--', '-.', ':']
    n_plots = 1 + int(u is not None) + int(ys is not None)

    if n_plots > 1:
        fig, axs = plt.subplots(
            n_plots, 1,
            figsize=(figsize_x, figsize_y),
            sharex=True
        )
    else:
        fig, axs = plt.subplots(
            figsize=(figsize_x, figsize_y),
            sharex=True
        )
        axs = [axs]

    for ax, vals, subplot_label, scale in zip(
            axs,
            [xs, ys] if ys is not None else [xs],
            [x_subplot_label, y_subplot_label],
            [x_scale, y_scale]
    ):
        for i, (x, label) in enumerate(zip(vals, labels)):
            ax.plot(
                t, x,
                label=label,
                zorder=i,
                color='k' if i == 0 else ibm_available_colors[i - 1],
                alpha=0.75,
                linestyle=available_lines[i % len(available_lines)]
            )

        if legend_outside_plots:
            ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=len(labels))
        else:
            ax.legend()
        ax.margins(x=0)
        if scientific_notation:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        if subplot_label is not None:
            ax.set_title(subplot_label)
        if scale is not None:
            ax.set_ylabel(scale)

    if u is not None:
        ax3 = axs[-1]
        ax3.plot(t, u, label='u', color='k')
        if legend_outside_plots:
            ax3.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center')
        else:
            ax3.legend()
        ax3.margins(x=0)
        if scientific_notation:
            ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax3.set_title(u_subplot_label)
        if u_scale is not None:
            ax3.set_ylabel(u_scale)

    if time_scale is not None:
        axs[-1].set_xlabel(time_scale)

    return fig
