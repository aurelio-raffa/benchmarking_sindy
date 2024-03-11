import os

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import font_manager

from src.utils import root_path

# set seaborn style
sns.set_style('whitegrid')

# prepare fonts
font_dirs = [os.path.join(root_path, 'data/fonts/inter/')]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


def set_font(publish: bool = bool(os.getenv('PUBLISH'))):
    if publish:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        plt.rcParams['font.size'] = 11
    else:
        plt.rcParams['font.family'] = 'Inter'
        plt.rcParams['font.size'] = 11
