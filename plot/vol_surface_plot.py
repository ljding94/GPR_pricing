import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from SVI_vol import calc_raw_SVI_skew_T1, calc_raw_SVI_surface, local_var, local_vol2
from scipy.optimize import minimize
from scipy.interpolate import griddata


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
    # ax6.yaxis.tick_right()
    ax6.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax6.xaxis.set_minor_locator(plt.MultipleLocator(0.25))

    # add annotations
    annot = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$"]
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        ax.text(0.7, 0.8, annot[i], transform=ax.transAxes, fontsize=9)

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
    # Plot the colored surface without mesh lines
    surface = ax1.plot_surface(k_grid, T_grid, w_vals_ij.T, cmap="rainbow", edgecolor="none", antialiased=True)
    # Plot the mesh as a wireframe overlay
    mesh = ax1.plot_wireframe(k_grid, T_grid, w_vals_ij.T, rstride=15, cstride=15, color="gray", linewidth=0.1)

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
    ax1.view_init(elev=25, azim=-120)

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
    surf = ax2.plot_surface(S_grid, t_grid, vol2_vals.T, cmap="rainbow", antialiased=True, edgecolor="none")
    mesh = ax2.plot_wireframe(S_grid, t_grid, vol2_vals.T, rstride=15, cstride=15, color="gray", linewidth=0.1)
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
    ax2.view_init(elev=38, azim=-60)

    ax1.text2D(0.8, 0.8, r"$(a)$", transform=ax1.transAxes, fontsize=9)
    ax2.text2D(0.8, 0.8, r"$(b)$", transform=ax2.transAxes, fontsize=9)

    plt.tight_layout(pad=1.5, w_pad=-2)
    plt.savefig("./figures/svi_surface.pdf", format="pdf")
    plt.savefig("./figures/svi_surface.png", dpi=300)
    plt.show()
    plt.close()


def try_to_fit_SVI_surface(tex_lw=240.71031, ppi=72):
    # 1 read real data
    # data = pd.read_csv("../data/data_test/w_grid_data_2023-10-04.csv")
    # k = data['k_grid'].values
    # T = data['T_grid'].values
    # w = data['total_var_grid'].values

    data = pd.read_csv("../data/data_test/volatility_surface_2023-10-04.csv")
    k = data["LOG_MONEYNESS"].values  # log-moneyness
    T = data["YTE"].values
    w = data["TOTAL_VARIANCE"].values

    # Filter data by time to maturity constraints
    mask = (T > 0.2) & (T < 1.0)
    k_2fit = k[mask]
    T_2fit = T[mask]
    w_2fit = w[mask]

    print(f"Filtered data points: {len(k)} (from original dataset)")

    # Define objective function for SVI surface fitting
    def objective_svi_surface(params, k_data, T_data, w_data):
        a1, b, rho, m, sigma, lam = params

        # Calculate predicted variance for each data point
        w_pred = np.zeros_like(w_data)
        for i in range(len(k_data)):
            w_pred[i] = calc_raw_SVI_surface(k_data[i], T_data[i], a1, b, rho, m, sigma, lam)

        # Return mean squared error
        return np.mean((w_data - w_pred) ** 2)

    # Initial guess for SVI parameters
    initial_params = [0.01, 0.15, 0.2, 0.2, 0.5, 0.5]

    # Parameter bounds (a1>=0, b>=0, -1<=rho<=1, m real, sigma>0, lam real)
    bounds = [(0, 1), (0, 2), (-0.99, 0.99), (-2, 2), (0.01, 2), (-2, 2)]

    # Perform optimization
    result = minimize(objective_svi_surface, initial_params, args=(k_2fit, T_2fit, w_2fit), method="L-BFGS-B", bounds=bounds)

    # Extract fitted parameters
    a1_fit, b_fit, rho_fit, m_fit, sigma_fit, lam_fit = result.x

    print(f"Fitted SVI parameters:")
    print(f"a1 = {a1_fit:.6f}")
    print(f"b = {b_fit:.6f}")
    print(f"rho = {rho_fit:.6f}")
    print(f"m = {m_fit:.6f}")
    print(f"sigma = {sigma_fit:.6f}")
    print(f"lambda = {lam_fit:.6f}")
    print(f"Final MSE: {result.fun:.8f}")

    # Calculate fitted values for comparison
    w_fitted = np.zeros_like(w)
    for i in range(len(k)):
        w_fitted[i] = calc_raw_SVI_surface(k[i], T[i], a1_fit, b_fit, rho_fit, m_fit, sigma_fit, lam_fit)

    # Create 3D scatter plot of the variance surface
    # fig = plt.figure(figsize=(12, 8))
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.8), dpi=ppi)
    ax = fig.add_subplot(111, projection="3d")

    # Create scatter plot with color mapping based on variance values
    scatter = ax.scatter(k_2fit, T_2fit, w_2fit, c=w_2fit, cmap="rainbow", s=3, alpha=0.6, label="Market observed")
    # Also plot fitted surface
    k_grid_plot = np.linspace(k_2fit.min(), k_2fit.max(), 50)
    T_grid_plot = np.linspace(T_2fit.min(), T_2fit.max(), 50)
    K_mesh, T_mesh = np.meshgrid(k_grid_plot, T_grid_plot)

    # Calculate fitted surface values
    w_surface = np.zeros_like(K_mesh)
    for i in range(len(k_grid_plot)):
        for j in range(len(T_grid_plot)):
            w_surface[j, i] = calc_raw_SVI_surface(K_mesh[j, i], T_mesh[j, i], a1_fit, b_fit, rho_fit, m_fit, sigma_fit, lam_fit)

    # Plot fitted surface
    ax.plot_surface(K_mesh, T_mesh, w_surface, alpha=0.3, color="gray", linewidth=0.5, label="SVI fitted")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label("Total Variance", fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))

    ax.tick_params(which="both", direction="in", top="on", right="on", labelsize=7, pad=-3)

    # Set labels and title
    ax.set_xlabel(r"$k$", fontsize=9, labelpad=-7)
    ax.set_ylabel(r"$T$", fontsize=9, labelpad=-7)
    ax.set_zlabel(r"$w$", fontsize=9, labelpad=-7)
    ax.set_title("SPX as of 2023-10-04", fontsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper left", markerscale=1.5)

    # Adjust viewing angle for better visualization
    ax.view_init(elev=20, azim=45)

    plt.tight_layout(pad=1.5)
    plt.savefig("./figures/svi_surface_fit.pdf", format="pdf")
    plt.savefig("./figures/svi_surface_fit.png", dpi=300)
    plt.show()


    # Create heatmap plot of the fitted volatility surface
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.6), dpi=ppi, sharex=True, sharey=True)

    # Create finer grid for heatmap
    k_heatmap = np.linspace(k_2fit.min(), k_2fit.max(), 50)
    T_heatmap = np.linspace(T_2fit.min(), T_2fit.max(), 50)
    K_heat, T_heat = np.meshgrid(k_heatmap, T_heatmap)

    # Calculate fitted surface for heatmap
    w_heatmap = np.zeros_like(K_heat)
    for i in range(len(k_heatmap)):
        for j in range(len(T_heatmap)):
            w_heatmap[j, i] = calc_raw_SVI_surface(K_heat[j, i], T_heat[j, i], a1_fit, b_fit, rho_fit, m_fit, sigma_fit, lam_fit)

    # Plot heatmap of fitted surface
    im1 = ax1.imshow(w_heatmap, cmap='rainbow', aspect='auto', origin='lower',
                    extent=[k_heatmap.min(), k_heatmap.max(), T_heatmap.min(), T_heatmap.max()])
    ax1.set_xlabel(r'$k$', fontsize=9, labelpad=0)
    ax1.set_ylabel(r'$T$', fontsize=9, labelpad=0)
    ax1.tick_params(which='both', direction='in', top='on', right='on', labelsize=7)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # Add colorbar for fitted surface
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("top", size="3%", pad=0.03)
    cb1 = plt.colorbar(im1, cax=cax1, orientation="horizontal")
    cb1.set_label(r'$w$', fontsize=7)
    cb1.ax.tick_params(labelsize=7)
    cax1.xaxis.set_ticks_position("top")
    cax1.xaxis.set_label_position("top")
    cax1.xaxis.set_major_locator(plt.MultipleLocator(0.04))

    # Plot heatmap of residuals (observed - fitted)
    # First, interpolate observed data to grid
    w_observed_grid = griddata((k_2fit, T_2fit), w_2fit, (K_heat, T_heat), method='linear')
    residuals = w_observed_grid - w_heatmap

    im2 = ax2.imshow(residuals, cmap='RdBu_r', aspect='auto', origin='lower',
                    extent=[k_heatmap.min(), k_heatmap.max(), T_heatmap.min(), T_heatmap.max()],
                    vmin=-np.nanmax(np.abs(residuals)), vmax=np.nanmax(np.abs(residuals)))
    ax2.set_xlabel(r'$k$', fontsize=9, labelpad=0)
    #ax2.set_ylabel(r'$T$', fontsize=9)
    ax2.tick_params(which='both', direction='in', top='on', right='on', labelsize=7)

    # Add colorbar for residuals
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("top", size="3%", pad=0.03)
    cb2 = plt.colorbar(im2, cax=cax2, orientation="horizontal")
    cb2.set_label(r'$\Delta w$', fontsize=7)
    cb2.ax.tick_params(labelsize=7)
    cax2.xaxis.set_ticks_position("top")
    cax2.xaxis.set_label_position("top")

    plt.tight_layout(pad=0.2)
    plt.savefig("./figures/svi_surface_heatmap.pdf", format="pdf")
    plt.savefig("./figures/svi_surface_heatmap.png", dpi=300)
    plt.show()
    plt.close()
