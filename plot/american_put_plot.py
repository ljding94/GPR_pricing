import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_american_put_GPR_fitting(tex_lw=240.71031, ppi=72):
    pass
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.8), dpi=ppi)
    ax1 = plt.subplot(231)