import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from SVI_vol import calc_raw_SVI_skew_T1, calc_raw_SVI_surface, local_var, local_vol2


def plot_Kvar_versus_SVI_params(tex_lw=240.71031, ppi=72):
    # baseline SVI parameters
    a1, b, rho, m, sigma = 0.005, 0.1, 0.2, 0.3, 0.4
    lam = 0.5
    k_vals = np.linspace(-0.15, 0.15, 100)

    # create figure and subplots
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.5), dpi=ppi)


