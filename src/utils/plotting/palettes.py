from matplotlib.colors import LinearSegmentedColormap


from __init__ import *


# definition of design libraries
ibm_design_library = {
    'aqua': '#31C9B0',
    'light_blue': '#648FFF',
    'purple': '#785EF0',
    'hot_pink': '#DC267F',
    'orange': '#FE6100',
    'yellow': '#FFB000',
}
tol = {
    'indigo': '#332288',
    'green': '#117733',
    # 'aqua': '#44AA99',
    'lightblue': '#88CCEE',
    'sand': '#DDCC77',
    'rose': '#CC6677',
    'orchid': '#AA4499',
    'violetred': '#882255'
}
cdict = {
    'red': [
        [0.0, 1.0, 1.0],
        [1.0, 96 / 255, 96 / 255]
    ],
    'green': [
        [0.0, 1.0, 1.0],
        [1.0, 124 / 255, 124 / 255]
    ],
    'blue': [
        [0.0, 1.0, 1.0],
        [1.0, 148 / 255, 148 / 255]
    ]
}
poliblue_cmap = LinearSegmentedColormap('PoliBlue', segmentdata=cdict, N=256)
markers = ['o', 's', 'v', 'P', 'h', 'X', '^', '<', '>']
ibm_available_colors = [
    ibm_design_library['orange'],
    ibm_design_library['purple'],
    ibm_design_library['light_blue'],
    ibm_design_library['hot_pink'],
    ibm_design_library['aqua'],
    ibm_design_library['yellow']
]
