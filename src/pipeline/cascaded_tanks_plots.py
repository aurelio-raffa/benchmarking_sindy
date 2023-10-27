import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import r2_score

sns.set_style('whitegrid')

from __init__ import root_path

from src.utils.plotting.palettes import ibm_design_library, tol

colors = ibm_design_library.copy()
colors.update(tol)
color_keys = list(colors.keys())
color_vals = list(colors.values())

if __name__ == '__main__':
    data = pd.read_csv(os.path.join(root_path, 'data/Benchmarks/cascaded_tanks.csv'))
    data = pd.concat([data, pd.read_csv(os.path.join(root_path, 'outputs/SINDy_bad_library.csv'))], axis=1)
    data.rename(columns={'x': 'x_bad'}, inplace=True)
    print(data)
    data = data.merge(pd.read_csv(os.path.join(root_path, 'outputs/SINDy_good_library.csv')), on='time')
    data.rename(columns={
        'SINDy_naif': 'SINDy, uninformed',
        'SINDy_prior': 'SINDy, informed',
        'x_bad': 'SINDy + hidden, wrong library',
        'x': 'SINDy + hidden, informed library',
        'true': 'Test trajectory',
        'time': 'Time [s]'
    }, inplace=True)
    data = data.loc[:, ['Time [s]', 'Test trajectory', 'ARX', 'SINDy, uninformed', 'SINDy, informed', 'SINDy + hidden, wrong library', 'SINDy + hidden, informed library']]

    for col in ['ARX', 'SINDy, uninformed', 'SINDy, informed', 'SINDy + hidden, wrong library', 'SINDy + hidden, informed library']:
        non_na = ~data.loc[:, col].isna()
        print(col, r2_score(data.loc[non_na, 'Test trajectory'], data.loc[non_na, col]))


    data = data.melt(id_vars='Time [s]').rename(columns={'value': 'Output', 'variable': 'Signal'})

    sns.lineplot(data=data, x='Time [s]', y='Output', hue='Signal', palette={
        'Test trajectory': 'k',
        'ARX': color_vals[0] + 'C0',
        'SINDy, uninformed': color_vals[1] + 'C0',
        'SINDy, informed': color_vals[2] + 'C0',
        'SINDy + hidden, wrong library': color_vals[3] + 'C0',
        'SINDy + hidden, informed library': color_vals[4] + 'C0'
    })
    plt.show()
