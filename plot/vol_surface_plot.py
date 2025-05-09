import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from SVI_vol import calc_raw_SVI_skew_T1, calc_raw_SVI_surface, local_var, local_vol2


def plot_illustrate_svi_curve(tex_lw=240.71031, ppi=72):
    # baseline SVI parameters
    a1, b, rho, m, sigma = 0.01, 0.15, 0.2, 0.2, 0.5
    k_vals = np.linspace(-0.2, 0.2, 100)

    # create figure and subplots
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 1), dpi=ppi)
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(232, sharey=ax1, sharex=ax1)
    ax3 = plt.subplot(233, sharey=ax1, sharex=ax1)
    ax4 = plt.subplot(234, sharey=ax1, sharex=ax1)
    ax5 = plt.subplot(235, sharey=ax1, sharex=ax1)
    ax6 = plt.subplot(236, sharey=ax1)

    colormap = plt.get_cmap("rainbow")
    base_params = dict(a1=a1, b=b, rho=rho, m=m, sigma=sigma)
    configs = [
        (ax1, "a1", np.arange(0.00, 0.0201, 0.002)),
        (ax2, "b", np.arange(0.0, 0.31, 0.03)),
        (ax3, "rho", np.arange(-0.4, 0.81, 0.2)),
        (ax4, "m", np.arange(-0.2, 0.61, 0.04)),
        (ax5, "sigma", np.arange(0.00, 1.01, 0.1)),
    ]
    tex = {"a1": r"$a'$", "b": r"$b$", "rho": r"$\rho$", "m": r"$m$", "sigma": r"$\sigma$"}

    for ax, name, vals in configs:
        colors = colormap(np.linspace(0, 1, len(vals)))
        for i, v in enumerate(vals):
            p = base_params.copy()
            p[name] = v
            w_vals = calc_raw_SVI_skew_T1(k_vals, p["a1"], p["b"], p["rho"], p["m"], p["sigma"])
            ax.plot(k_vals, w_vals, lw=1, color=colors[i])

        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="3%", pad=0.03)
        cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(tex[name], fontsize=7, labelpad=1.5)
        # shift the colorbar label to the right end
        # cax.xaxis.set_label_coords(0.9, 2.5)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cax.xaxis.set_tick_params(which="both", direction="in", top="on", right="on", labelsize=7, pad=0)
        start, end = vals.min(), vals.max()
        mid = (start + end) / 2
        cax.set_xticks([start, mid, end])

        ax.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=False, labelsize=7)
        ax.set_xlabel(r"$k$", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$w(k; \chi_R)$", fontsize=9, labelpad=0)
    ax4.set_ylabel(r"$w(k; \chi_R)$", fontsize=9, labelpad=0)
    ax1.tick_params(labelleft=True)
    ax4.tick_params(labelleft=True)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    w0 = calc_raw_SVI_skew_T1(0, a1, b, rho, m, sigma)
    T = np.linspace(0.01, 1.0, 100)
    lams = np.arange(0.0, 1.01, 0.2)
    colors = colormap(np.linspace(0, 1, len(lams)))
    for i, lam in enumerate(lams):
        fT = T * np.exp(lam * (1 - T))
        ax6.plot(T, fT * w0, lw=1, color=colors[i], label=f"{lam:.2f}")
    norm = Normalize(vmin=lams.min(), vmax=lams.max())
    sm = ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("top", size="3%", pad=0.03)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(r"$\lambda$", fontsize=7, labelpad=1.5)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cax.xaxis.set_tick_params(which="both", direction="in", top="on", right="on", labelsize=7, pad=0)
    start, end = lams.min(), lams.max()
    mid = (start + end) / 2
    cax.set_xticks([start, mid, end])
    ax6.yaxis.set_label_position("right")
    ax6.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=False, labelsize=7)
    ax6.set_xlabel(r"$T$", fontsize=9, labelpad=0)
    ax6.set_ylabel(r"$w(k=0;\chi'_R)$", fontsize=9, labelpad=0)
    #ax6.yaxis.tick_right()
    ax6.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax6.xaxis.set_minor_locator(plt.MultipleLocator(0.25))

    # add annotations
    annot = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$"]
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        ax.text(0.7,0.8, annot[i], transform=ax.transAxes, fontsize=9)


    # finalize and save
    plt.tight_layout(pad=0.2)
    plt.savefig("./figures/svi_curve.pdf", format="pdf")
    plt.savefig("./figures/svi_curve.png", dpi=300)
    plt.show()
    plt.close()


def plot_illustrative_vol_surface_local_vol(tex_lw=240.71031, ppi=72):
    # baseline SVI parameters
    a1, b, rho, m, sigma = 0.01, 0.15, 0.2, 0.2, 0.5
    lam = 0.5
    k_vals = np.linspace(-0.15, 0.15, 100)

    # create figure and subplots
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.5), dpi=ppi)
    ax1 = plt.subplot(121, projection="3d")
    ax2 = plt.subplot(122, projection="3d")

    T_vals = np.linspace(0.01, 1.0, 100)

    w_vals_ij = np.zeros((len(k_vals), len(T_vals)))
    for i, k in enumerate(k_vals):
        for j, T in enumerate(T_vals):
            w_vals_ij[i, j] = calc_raw_SVI_surface(k, T, a1, b, rho, m, sigma, lam)
    # Create meshgrid for 3D plot
    k_grid, T_grid = np.meshgrid(k_vals, T_vals)
    surf = ax1.plot_surface(k_grid, T_grid, w_vals_ij.T, cmap="rainbow", linewidth=0, antialiased=True)

    ax1.set_xlabel(r"$k$", fontsize=9, labelpad=-8)
    ax1.set_ylabel(r"$T$", fontsize=9, labelpad=-8)
    ax1.set_zlabel(r"$w$", fontsize=9, labelpad=-6)
    ax1.tick_params(which="both", direction="in", top="on", right="on", labelsize=7, pad=-4)
    ax1.tick_params(axis="z", which="both", direction="in", top="on", right="on", labelsize=7, pad=-2)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax1.zaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax1.zaxis.set_minor_locator(plt.MultipleLocator(0.005))
    ax1.set_zlim(0, 0.02)
    ax1.view_init(elev=20, azim=-120)

    S_vals = np.linspace(0.9, 1.1, 100)
    t_vals = np.linspace(0.01, 1.0, 100)
    S0 = 1.0
    r = 0.05

    vol2_vals = np.zeros((len(S_vals), len(t_vals)))
    for i, k in enumerate(S_vals):
        for j, T in enumerate(t_vals):
            vol2_vals[i, j] = local_vol2(S0, k, T, r, a1, b, rho, m, sigma, lam)
    # Create meshgrid for 3D plot
    S_grid, t_grid = np.meshgrid(S_vals, t_vals)
    surf = ax2.plot_surface(S_grid, t_grid, vol2_vals.T, cmap="rainbow", linewidth=0, antialiased=True)
    ax2.set_xlabel(r"$S$", fontsize=9, labelpad=-8)
    ax2.set_ylabel(r"$t$", fontsize=9, labelpad=-8)
    ax2.set_zlabel(r"$\sigma^2_{loc}$", fontsize=9, labelpad=-6)
    ax2.tick_params(which="both", direction="in", top="on", right="on", labelsize=7, pad=-4)
    ax2.tick_params(axis="z", which="both", direction="in", top="on", right="on", labelsize=7, pad=-2)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax2.zaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax2.zaxis.set_minor_locator(plt.MultipleLocator(0.005))
    # ax2.set_zlim(0, 0.02)
    ax2.view_init(elev=20, azim=-60)

    ax1.text2D(0.8,0.8, r"$(a)$", transform=ax1.transAxes, fontsize=9)
    ax2.text2D(0.8,0.8, r"$(b)$", transform=ax2.transAxes, fontsize=9)

    plt.tight_layout(pad=1.5, w_pad=-2)
    plt.savefig("./figures/svi_surface.pdf", format="pdf")
    plt.savefig("./figures/svi_surface.png", dpi=300)
    plt.show()
    plt.close()
