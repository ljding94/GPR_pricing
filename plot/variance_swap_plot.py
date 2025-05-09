import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from SVI_vol import calc_raw_SVI_skew_T1, calc_raw_SVI_surface, local_var, local_vol2
from variance_swap import variace_swap_strike


def plot_Kvar_versus_SVI_params(tex_lw=240.71031, ppi=72):
    # baseline SVI parameters
    a1, b, rho, m, sigma = 0.01, 0.15, 0.2, 0.2, 0.5
    r = 0.03

    # create figure and subplots/
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.8), dpi=ppi)
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(232, sharey=ax1)
    ax3 = plt.subplot(233, sharey=ax1)
    ax4 = plt.subplot(234, sharey=ax1)
    ax5 = plt.subplot(235, sharey=ax1)
    ax6 = plt.subplot(236, sharey=ax1)

    n = 20
    param_configs = [
        (ax1, "a1", r"$a'$", np.linspace(0.00, 0.02, n), 0.01),
        (ax2, "b", r"$b$", np.linspace(0.0, 0.3, n), 0.1),
        (ax3, "rho", r"$\rho$", np.linspace(-0.4, 0.8, n), 0.4),
        (ax4, "m", r"$m$", np.linspace(-0.2, 0.6, n), 0.4),
        (ax5, "sigma", r"$\sigma$", np.linspace(0.00, 1.0, n), 0.5),
        (ax6, "r", r"$r$", np.linspace(0.00, 0.06, n), 0.03),
    ]
    tex = {
        "a1": r"$a'$",
        "b": r"$b$",
        "rho": r"$\rho$",
        "m": r"$m$",
        "sigma": r"$\sigma$",
        "r": r"$r$",
    }
    base_params = dict(a1=a1, b=b, rho=rho, m=m, sigma=sigma, r=r)
    for ax, name, tex, vals, mlocator in param_configs:
        Kvar_vals = []
        for i, v in enumerate(vals):
            p = base_params.copy()
            p[name] = v
            print(p)
            Kvar = variace_swap_strike(p["a1"], p["b"], p["rho"], p["m"], p["sigma"], 1, 1, p["r"])
            Kvar_vals.append(Kvar)
        ax.plot(vals, Kvar_vals, lw=1, color="royalblue")
        ax.set_xlabel(tex, fontsize=9, labelpad=0)
        ax.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=False, labelsize=7, pad=1)
        ax.xaxis.set_major_locator(plt.MultipleLocator(mlocator))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(mlocator * 0.5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.005))
    ax1.tick_params(labelleft=True)
    ax4.tick_params(labelleft=True)
    ax1.set_ylabel(r"$K_{var}$", fontsize=9, labelpad=0)
    ax4.set_ylabel(r"$K_{var}$", fontsize=9, labelpad=0)

    # add annotations
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$"]
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        x = 0.7
        y = 0.8
        ax.text(x, y, annos[i], transform=ax.transAxes, fontsize=9)

    plt.tight_layout(pad=0.2)
    plt.savefig("./figures/Kvar_vs_params.png", dpi=ppi)
    plt.savefig("./figures/Kvar_vs_params.pdf", format="pdf")
    plt.show()
    plt.close()


def plot_variance_swap_GPR_fitting(tex_lw=240.71031, ppi=72):
    folder = "../data/20250505_vs"
    # create figure and subplots
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.5), dpi=ppi)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    # get LML data
    """
    save_dict = {"LML": LML, "theta_opt": theta_opt, "LML_opt": lml_opt}
        # add each grid array
    for name, grid in zip(param_names, grids):
        save_dict[f"{name}_grid"] = grid
    """
    data = np.load(f"{folder}/variance_swap_Kvar_LML.npz")
    print("data", data.files)

    LML = data["LML"]
    theta_opt = data["theta_opt"]
    LML_opt = data["LML_opt"]

    # for variance swap, grid is 1D
    l_grid = data["length_scale_grid"]

    ax1.plot(l_grid, LML, lw=1, color="royalblue")
    ax1.plot(theta_opt[0], LML_opt, "rx", markersize=8)
    ax1.set_xlabel(r"$l_g$", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$LML$", fontsize=9, labelpad=0)
    ax1.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7, pad=2)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(2.5))
    ax1.text(0.2, 0.6, r"$K_{var}$", fontsize=9, transform=ax1.transAxes)

    # get prediction data
    data = np.loadtxt("../data/20250505_vs/variance_swap_Kvar_prediction.txt", skiprows=1, delimiter=",")
    Kvar, ML_predicted = data[:, 0], data[:, 1]

    ax2.scatter(Kvar, ML_predicted, marker=".", s=2, facecolor="none", edgecolor="black")
    #Err = 100*np.abs(ML_predicted - Kvar)/np.maximum(Kvar, ML_predicted)
    scale = np.mean(np.abs(Kvar))  # or np.median(np.abs(Y)), or (Y.max()-Y.min())
    Err = 100 * np.abs(ML_predicted - Kvar) / scale
    Err = np.mean(Err)
    ax2.text(0.5, 0.3, rf"Err={Err:.1f}%", fontsize=9, transform=ax2.transAxes)
    ax2.grid()
    ax2.set_xlabel(r"Ground Truth", fontsize=9, labelpad=0)
    ax2.set_ylabel(r"ML Predicted", fontsize=9, labelpad=0)
    ax2.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7, pad=2)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.025))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.025))
    ax2.set_xlim(-0.01, )
    ax2.set_ylim(-0.01, )
    ax2.text(0.2, 0.6, r"$K_{var}$", fontsize=9, transform=ax2.transAxes)

    # add annotations
    ax1.text(0.7, 0.15, r"$(a)$", transform=ax1.transAxes, fontsize=9)
    ax2.text(0.7, 0.15, r"$(b)$", transform=ax2.transAxes, fontsize=9)

    plt.tight_layout(pad=0.2)
    plt.savefig("./figures/variance_swap_GPR_fitting.png", dpi=ppi)
    plt.savefig("./figures/variance_swap_GPR_fitting.pdf", format="pdf")
    plt.show()
    plt.close()
