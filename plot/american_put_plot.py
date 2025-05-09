import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
from SVI_vol import bs_price
from American_put import price_american_option_PSOR


def plot_american_price_solution(tex_lw=240.71031, ppi=72):
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.8), dpi=ppi)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222, sharex=ax1)
    ax3 = plt.subplot(223, sharex=ax1)
    ax4 = plt.subplot(224, sharex=ax1)

    a1, b, rho, m, sigma = 0.03, 0.0, 0.0, 0.001, 0.0
    r = 0.05
    K = 1.0
    T = 1.0
    lam = 0.0
    S0 = 1
    ni, nf = 70, 130
    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V = price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="call", refine_theta=False)
    ax1.plot(S[ni:nf], price_grid[ni:nf], label="A.C.", lw=1, color="royalblue")
    ax2.plot(S[ni:nf], delta_grid[ni:nf], label="A.C.", lw=1, color="royalblue")
    ax3.plot(S[ni:nf], gamma_grid[ni:nf], label="A.C.", lw=1, color="royalblue")
    ax4.plot(S[ni:nf], theta_grid[ni:nf], label="A.C.", lw=1, color="royalblue")

    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V = price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put", refine_theta=False)
    ax1.plot(S[ni:nf], price_grid[ni:nf], label="A.P.", lw=1, color="tomato")
    ax2.plot(S[ni:nf], delta_grid[ni:nf], label="A.P.", lw=1, color="tomato")
    ax3.plot(S[ni:nf], gamma_grid[ni:nf], label="A.P.", lw=1, color="tomato")
    ax4.plot(S[ni:nf], theta_grid[ni:nf], label="A.P.", lw=1, color="tomato")

    S_bs = S[ni:nf]
    bs_price_call, bs_delta_call, bs_gamma_call, bs_theta_call = bs_price(S_bs, K, T, r, np.sqrt(a1), type="call")
    ax1.plot(S_bs, bs_price_call, label="E.C.", linestyle=(-2, (2, 3)), lw=1, color="black")
    ax2.plot(S_bs, bs_delta_call, label="E.C.", linestyle=(-2, (2, 3)), lw=1, color="black")
    ax3.plot(S_bs, bs_gamma_call, label="E.C.", linestyle=(-2, (2, 3)), lw=1, color="black")
    ax4.plot(S_bs, bs_theta_call, label="E.C.", linestyle=(-2, (2, 3)), lw=1, color="black")
    bs_price_put, bs_delta_put, bs_gamma_put, bs_theta_put = bs_price(S_bs, K, T, r, np.sqrt(a1), type="put")
    ax1.plot(S_bs, bs_price_put, label="E.P.", linestyle=(0, (1, 4)), lw=1, color="black")
    ax2.plot(S_bs, bs_delta_put, label="E.P.", linestyle=(0, (1, 4)), lw=1, color="black")
    ax3.plot(S_bs, bs_gamma_put, label="E.P.", linestyle=(0, (1, 4)), lw=1, color="black")
    ax4.plot(S_bs, bs_theta_put, label="E.P.", linestyle=(0, (1, 4)), lw=1, color="black")

    # ax1.set_xlabel(r"$S$")
    ax1.set_ylabel(r"$V$", fontsize=9, labelpad=0)
    # ax2.set_xlabel(r"$S$")
    ax2.set_ylabel(r"$\Delta$", fontsize=9, labelpad=0)
    ax3.set_xlabel(r"$S$", fontsize=9, labelpad=0)
    ax3.set_ylabel(r"$\Gamma$", fontsize=9, labelpad=0)
    ax4.set_xlabel(r"$S$", fontsize=9, labelpad=0)
    ax4.set_ylabel(r"$\Theta$", fontsize=9, labelpad=-3)

    ax1.legend(ncol=1, columnspacing=0.5, handlelength=1.0, handletextpad=0.2, labelspacing=0.2, frameon=False, fontsize=9)
    ax3.legend(ncol=1, columnspacing=0.5, handlelength=1.0, handletextpad=0.2, labelspacing=0.2, frameon=False, fontsize=9)
    mlocator = [0.1, 1.0, 2, 0.04]

    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=True, labelsize=7, pad=2)
        ax.yaxis.set_major_locator(plt.MultipleLocator(mlocator[i]))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(mlocator[i] * 0.5))
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax3.tick_params(labelbottom=True)
    ax4.tick_params(labelbottom=True)

    # annotations
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$"]
    for i, ax in enumerate([ax1, ax2, ax3, ax4]):
        ax.text(0.8, 0.25, annos[i], transform=ax.transAxes, fontsize=9)

    plt.tight_layout(pad=0.2)
    plt.savefig("./figures/american_price_solution.png", dpi=ppi)
    plt.savefig("./figures/american_price_solution.pdf", format="pdf")
    plt.show()
    plt.close()


def plot_american_price_solution_per_r(tex_lw=240.71031, ppi=72):
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 1.4), dpi=ppi)

    ax11 = plt.subplot(321)
    ax12 = plt.subplot(322, sharex=ax11, sharey=ax11)
    ax21 = plt.subplot(323, sharex=ax11, sharey=ax11)
    ax22 = plt.subplot(324, sharex=ax11, sharey=ax11)
    ax31 = plt.subplot(325, sharex=ax11)
    ax32 = plt.subplot(326, sharex=ax11)


    a1, b, rho, m, sigma = 0.03, 0.0, 0.0, 0.001, 0.0

    K = 1.0
    T = 1.0
    lam = 0.0
    S0 = 1


    Mrun, Nrun = 1000, 500
    ni, nf = 140, 280
    ntf = -50

    #Mrun, Nrun = 200, 100
    #ni,nf = 28, 56
    #ntf = -10

    Vmin, Vmax = 0.0, 0.3
    Gmin, Gmax = 0.0, 9

    rcolormap = plt.get_cmap("rainbow")
    rcolors = rcolormap(np.linspace(0, 1, 3))

    r = 0.10
    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V, all_Gamma = price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, M= Mrun, N=Nrun, option_type="put", refine_theta=False, all_Gamma=True)
    #Vmax = np.max(all_V[ni:nf, :])
    #Gmax = np.max(all_Gamma[ni:nf, :ntf])


    S_sub = S[ni:nf]
    t_vals = np.linspace(0, T, all_V.shape[1])
    S_mesh, t_mesh = np.meshgrid(S_sub, t_vals, indexing="ij")
    pcm11 = ax11.pcolormesh(S_mesh, t_mesh, all_V[ni:nf, :], shading="auto", cmap="rainbow_r", vmin=Vmin, vmax=Vmax, rasterized=True)
    #ax11.set_title(r"$V$", fontsize=9, pad=0)
    cbar11 = plt.colorbar(pcm11, ax=ax11, fraction=0.04, pad=0.02, orientation="horizontal",location="top")
    cbar11.ax.xaxis.set_ticks_position("top")
    cbar11.ax.xaxis.set_label_position("top")
    cbar11.set_label(r"$V$", fontsize=9, labelpad=0)
    cbar11.ax.tick_params(which="both", direction="in", top="on", right="on", labelsize=7, pad=0)
    cbar11.ax.set_xticks([Vmin, 0.5*Vmax, Vmax])


    pcm12 = ax12.pcolormesh(S_mesh[:,:ntf], t_mesh[:,:ntf], all_Gamma[ni:nf, :ntf], shading="auto", cmap="rainbow_r", vmin=Gmin, vmax=Gmax, rasterized=True)
    cbar12 = plt.colorbar(pcm12, ax=ax12, fraction=0.04, pad=0.02, orientation="horizontal",location="top")
    cbar12.ax.xaxis.set_ticks_position("top")
    cbar12.ax.xaxis.set_label_position("top")
    cbar12.set_label(r"$\Gamma$", fontsize=9, labelpad=0)
    cbar12.ax.tick_params(which="both", direction="in", top="on", right="on", labelsize=7, pad=0)
    cbar12.ax.set_xticks([Gmin, 0.5*Gmax, Gmax])

    ax31.plot(S[ni:nf], price_grid[ni:nf], label=f"r={r:.2f}", lw=1, color=rcolors[0])
    ax32.plot(S[ni:nf], gamma_grid[ni:nf], label=f"r={r:.2f}", lw=1, color=rcolors[0])

    r = 0.05
    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V, all_Gamma = price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, M= Mrun, N=Nrun, option_type="put", refine_theta=False, all_Gamma=True)
    ax31.plot(S[ni:nf], price_grid[ni:nf], label=f"r={r:.2f}", lw=1, color=rcolors[1])
    ax32.plot(S[ni:nf], gamma_grid[ni:nf], label=f"r={r:.2f}", lw=1, color=rcolors[1])

    r = 0.00
    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V, all_Gamma = price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, M= Mrun, N=Nrun, option_type="put", refine_theta=False, all_Gamma=True)
    S_sub = S[ni:nf]
    t_vals = np.linspace(0, T, all_V.shape[1])
    S_mesh, t_mesh = np.meshgrid(S_sub, t_vals, indexing="ij")

    pcm21 = ax21.pcolormesh(S_mesh, t_mesh, all_V[ni:nf, :], shading="auto", cmap="rainbow_r", vmin=Vmin, vmax=Vmax, rasterized=True)
    #ax21.set_title(r"$V$", fontsize=9, pad=0)
    #cbar21 = plt.colorbar(pcm21, ax=ax21, fraction=0.04, pad=0.02, orientation="horizontal",location="top")
    #cbar21.ax.xaxis.set_ticks_position("top")
    #cbar21.ax.xaxis.set_label_position("top")
    #cbar21.set_label(r"$V$", fontsize=9, labelpad=0)
    #cbar21.ax.tick_params(labelsize=7, direction="in")

    pcm22 = ax22.pcolormesh(S_mesh[:,:ntf], t_mesh[:,:ntf], all_Gamma[ni:nf, :ntf], shading="auto", cmap="rainbow_r", vmin=Gmin, vmax=Gmax, rasterized=True)
    #ax22.set_title(r"$\Gamma$", fontsize=9, pad=0)
    #cbar22 = plt.colorbar(pcm22, ax=ax22, fraction=0.03, pad=0.02, orientation="horizontal",location="top")
    #cbar22.ax.xaxis.set_ticks_position("top")
    #cbar22.ax.xaxis.set_label_position("top")
    #cbar22.set_label(r"$\Gamma$", fontsize=9, labelpad=0)
    #cbar22.ax.tick_params(labelsize=7, direction="in")

    ax31.plot(S[ni:nf], price_grid[ni:nf], label=f"r={r:.2f}", lw=1, color=rcolors[2])
    ax32.plot(S[ni:nf], gamma_grid[ni:nf], label=f"r={r:.2f}", lw=1, color=rcolors[2])



    S_bs = S[ni:nf]
    bs_price_put, bs_delta_put, bs_gamma_put, bs_theta_put = bs_price(S_bs, K, T, r, np.sqrt(a1), type="put")
    ax31.plot(S_bs, bs_price_put, label="E.P.", linestyle=(0, (2, 4)), lw=1, color="black")
    ax32.plot(S_bs, bs_gamma_put, label="E.P.", linestyle=(0, (2, 4)), lw=1, color="black")

    ax11.tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=True, labelsize=7, pad=2)
    ax11.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax11.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax11.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax11.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    ax21.tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=True, labelsize=7, pad=2)


    ax12.tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=False, labelsize=7, pad=2)
    ax22.tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=False, labelsize=7, pad=2)

    ax31.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7, pad=2)
    ax31.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax31.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax32.tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelright=True, labelleft=False, labelsize=7, pad=2)
    ax32.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax32.yaxis.set_minor_locator(plt.MultipleLocator(1))

    ax11.set_ylabel(r"$t$", fontsize=9, labelpad=0)
    ax21.set_ylabel(r"$t$", fontsize=9, labelpad=0)

    ax31.set_ylabel(r"$V$", fontsize=9, labelpad=0)
    ax31.set_xlabel(r"$S$", fontsize=9, labelpad=0)

    ax32.yaxis.set_label_position("right")
    ax32.set_ylabel(r"$\Gamma$", fontsize=9, labelpad=0)
    ax32.set_xlabel(r"$S$", fontsize=9, labelpad=0)

    ax31.legend(ncol=1, columnspacing=0.5, handlelength=1.0, handletextpad=0.2, labelspacing=0.2, frameon=False, fontsize=9)
    ax32.legend(ncol=1, columnspacing=0.5, handlelength=1.0, handletextpad=0.2, labelspacing=0.2, frameon=False, fontsize=9)

    ax11.text(0.5,0.4, r"$r=0.10$", transform=ax11.transAxes, fontsize=9)
    ax12.text(0.5,0.4, r"$r=0.10$", transform=ax12.transAxes, fontsize=9)
    ax21.text(0.5,0.4, r"$r=0.00$", transform=ax21.transAxes, fontsize=9)
    ax22.text(0.5,0.4, r"$r=0.00$", transform=ax22.transAxes, fontsize=9)

    # annotations
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$", r"$(e)$", r"$(f)$"]
    for i, ax in enumerate([ax11, ax12, ax21, ax22, ax31, ax32]):
        ax.text(0.8, 0.2, annos[i], transform=ax.transAxes, fontsize=9)

    print("Vmax", Vmax)
    print("Gmax", Gmax)

    plt.tight_layout(pad=0.2)
    plt.savefig("./figures/american_price_solution_per_r.png", dpi=300)
    plt.savefig("./figures/american_price_solution_per_r.pdf", format="pdf", dpi=300)
    plt.show()
    plt.close()




def get_sensitivity_data(folder, param):
    data = np.loadtxt(f"{folder}/american_put_{param}_sensitivity.csv", skiprows=1, delimiter=",", unpack=True)
    par, price, delta, gamma, theta = data
    return par, price, delta, gamma, theta


def plot_american_params_sensitivity(tex_lw=240.71031, ppi=72):

    folder = "../data/20250505"
    fig, axs = plt.subplots(4, 8, figsize=(tex_lw / ppi * 2.0, tex_lw / ppi * 1), dpi=ppi, sharex="col", sharey="row")

    params = ["a1", "b", "rho", "m", "sigma", "lam", "K", "r"]
    texs = [r"$a'$", r"$b$", r"$\rho$", r"$m$", r"$\sigma$", r"$\lambda$", r"$K$", r"$r$"]
    mlocator = [0.01, 0.2, 0.4, 0.4, 0.5, 0.5, 0.1, 0.05]
    ymlocator = [0.1, 0.5, 5, 0.02]
    for i in range(len(params)):
        par, price, delta, gamma, theta = get_sensitivity_data(folder, params[i])
        axs[0, i].plot(par, price, lw=1, color="royalblue")
        axs[1, i].plot(par, delta, lw=1, color="royalblue")
        color = "tomato" if i ==6 else "royalblue"
        axs[2, i].plot(par, gamma, lw=1, color=color)

        axs[3, i].plot(par, theta, lw=1, color="royalblue")

        for j in range(4):
            axs[j, i].tick_params(which="both", direction="in", top="on", right="on", labelbottom=False, labelleft=False, labelsize=7, pad=2)
            axs[j, 0].yaxis.set_major_locator(plt.MultipleLocator(ymlocator[j]))
            axs[j, 0].yaxis.set_minor_locator(plt.MultipleLocator(ymlocator[j] * 0.5))

        axs[3, i].xaxis.set_major_locator(plt.MultipleLocator(mlocator[i]))
        axs[3, i].xaxis.set_minor_locator(plt.MultipleLocator(mlocator[i] * 0.5))
        axs[3, i].set_xlabel(texs[i], fontsize=9, labelpad=0)
        axs[3, i].tick_params(labelbottom=True)

    axs[0, 0].set_ylabel(r"$V$", fontsize=9, labelpad=0)
    axs[0, 0].tick_params(labelleft=True)
    axs[1, 0].set_ylabel(r"$\Delta$", fontsize=9, labelpad=0)
    axs[1, 0].tick_params(labelleft=True)
    axs[2, 0].set_ylabel(r"$\Gamma$", fontsize=9, labelpad=0)
    axs[2, 0].tick_params(labelleft=True)
    axs[3, 0].set_ylabel(r"$\Theta$", fontsize=9, labelpad=0)
    axs[3, 0].tick_params(labelleft=True)

    # annotations
    alphabet = [r"$a$", r"$b$", r"$c$", r"$d$", r"$e$", r"$f$", r"$g$", r"$h$"]
    number = [r"$1$", r"$2$", r"$3$", r"$4$", r"$5$", r"$6$", r"$7$", r"$8$"]
    for i in range(len(params)):
        for j in range(4):
            axs[j, i].text(0.3, 0.7, r"$($" + alphabet[i] + number[j] + r"$)$", transform=axs[j, i].transAxes, fontsize=9)

    plt.tight_layout(pad=0.2)
    plt.savefig("./figures/american_params_sensitivity_price.png", dpi=ppi)
    plt.savefig("./figures/american_params_sensitivity_price.pdf", format="pdf")
    plt.show()
    plt.close()


def plot_american_put_LML(tex_lw=240.71031, ppi=72):
    folder = "../data/20250505"

    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 1), dpi=ppi)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    axs = [ax1, ax2, ax3, ax4]
    targets = ["price", "delta", "gamma", "theta"]
    texs = [r"$V$", r"$\Delta$", r"$\Gamma$", r"$\Theta$"]
    grid_size = 2
    ticks = {
        "price": {"length_scale": np.linspace(2.5, 3.5, grid_size), "noise_level": np.logspace(-3, -2, grid_size)},
        "delta": {"length_scale": np.linspace(2.0, 4.0, grid_size), "noise_level": np.logspace(-2, -1, grid_size)},
        "gamma": {"length_scale": np.linspace(1, 3, grid_size), "noise_level": np.logspace(-1, 0, grid_size)},
        "theta": {"length_scale": np.linspace(1.5, 3.0, grid_size), "noise_level": np.logspace(-3, -1, grid_size)},
    }
    ticklabels = {
        "price": {"noise_level": (r"$10^{-3}$", r"$10^{-2}$")},
        "delta": {"noise_level": (r"$10^{-2}$", r"$10^{-1}$")},
        "gamma": {"noise_level": (r"$10^{-1}$", r"$10^{0}$")},
        "theta": {"noise_level": (r"$10^{-3}$", r"$10^{-2}$")},
    }
    for i in range(len(targets)):
        data = np.load(f"{folder}/american_put_{targets[i]}_LML.npz")
        print("data", data.files)
        LML = data["LML"]
        theta_opt = data["theta_opt"]
        LML_opt = data["LML_opt"]
        l_grid = data["length_scale_grid"]
        sig_grid = data["noise_level_grid"]
        Xg, Yg = np.meshgrid(l_grid, sig_grid, indexing="ij")
        cs = axs[i].contour(Xg, Yg, LML, levels=200, cmap="viridis")
        axs[i].set_yscale("log")
        axs[i].plot(theta_opt[0], theta_opt[1], "rx", markersize=8)
        axs[i].tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7, pad=2)
        axs[i].text(0.5, 0.8, texs[i], transform=axs[i].transAxes, fontsize=9)
        axs[i].set_xlabel(r"$l_g$", fontsize=9, labelpad=0)
        axs[i].set_ylabel(r"$\sigma_g$", fontsize=9, labelpad=0)
        axs[i].set_xticks(ticks[targets[i]]["length_scale"])
        axs[i].set_yticks(ticks[targets[i]]["noise_level"])
        axs[i].set_yticklabels(ticklabels[targets[i]]["noise_level"], fontsize=7)
        print("ticks", ticks[targets[i]]["noise_level"])
        print("ticklabels", ticklabels[targets[i]]["noise_level"])
        axs[i].minorticks_off()


    # annotate
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$"]
    for i, ax in enumerate(axs):
        ax.text(0.7, 0.2, annos[i], transform=ax.transAxes, fontsize=9)

    plt.tight_layout(pad=0.2)
    plt.savefig("./figures/american_put_LML.png", dpi=ppi)
    plt.savefig("./figures/american_put_LML.pdf", format="pdf")
    plt.show()
    plt.close()


def plot_american_put_GPR_fitting(tex_lw=240.71031, ppi=72):
    folder = "../data/20250505"
    fig = plt.figure(figsize=(tex_lw / ppi * 1.0, tex_lw / ppi * 0.8), dpi=ppi)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    axs = [ax1, ax2, ax3, ax4]
    targets = ["price", "delta", "gamma", "theta"]
    texs = [r"$V$", r"$\Delta$", r"$\Gamma$", r"$\Theta$"]
    mlocator = [0.05, 0.5, 4, 0.04]
    for i in range(len(targets)):
        data = np.loadtxt(f"{folder}/american_put_{targets[i]}_prediction.txt", skiprows=1, unpack=True, delimiter=",")
        Y, Y_predict, Y_predict_err = data
        axs[i].scatter(Y, Y_predict,marker=".", s=2, color="royalblue", facecolor="none",edgecolor="black")
        scale = np.mean(np.abs(Y))  # or np.median(np.abs(Y)), or (Y.max()-Y.min())
        Err = 100 * np.abs(Y_predict - Y) / scale
        #Err = 100*np.abs(Y_predict - Y)/np.maximum(np.abs(Y), np.abs(Y_predict))
        Err = np.mean(Err)
        axs[i].text(0.95, 0.25, f"Err: {Err:.1f}%", transform=axs[i].transAxes, fontsize=9, ha="right")
        axs[i].grid()
        axs[i].text(0.2,0.7, texs[i], transform=axs[i].transAxes, fontsize=9)
        axs[i].yaxis.set_major_locator(plt.MultipleLocator(mlocator[i]))
        axs[i].yaxis.set_minor_locator(plt.MultipleLocator(mlocator[i] * 0.5))
        axs[i].xaxis.set_major_locator(plt.MultipleLocator(mlocator[i]))
        axs[i].xaxis.set_minor_locator(plt.MultipleLocator(mlocator[i] * 0.5))
        axs[i].tick_params(which="both", direction="in", top="on", right="on", labelbottom=True, labelleft=True, labelsize=7, pad=2)

    axall = fig.add_subplot(111, frameon=False)
    axall.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
    axall.set_xlabel(r"Ground Truth", fontsize=9, labelpad=-5)
    axall.set_ylabel(r"ML Predicted", fontsize=9, labelpad=-5)

    # annotation
    annos = [r"$(a)$", r"$(b)$", r"$(c)$", r"$(d)$"]
    for i, ax in enumerate(axs):
        ax.text(0.75, 0.1, annos[i], transform=ax.transAxes, fontsize=9)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0.12, bottom=0.12)
    plt.savefig("./figures/american_put_GPR_fitting.png", dpi=ppi)
    plt.savefig("./figures/american_put_GPR_fitting.pdf", format="pdf")
    plt.show()
    plt.close()