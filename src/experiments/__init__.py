import os
import sys

# we add to PATH the root folder
base_path = os.path.dirname(os.path.abspath(__file__))      # path to directory containing the script
root_path = os.path.dirname(os.path.dirname(base_path))     # path to directory containing sources (src)
sys.path.append(root_path)
