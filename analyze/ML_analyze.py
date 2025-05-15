import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import os
from scipy.optimize import curve_fit
import pickle
from scipy.spatial import cKDTree
from itertools import product
import time


def plot_data_distribution(targets_train, targets_test):
    print(targets_test)
    plt.figure(figsize=(10, 6))
    plt.hist(targets_train, bins=100, histtype="step", label="Train", density=True)
    plt.hist(targets_test, bins=100,  histtype="step",label="Test", density=True)
    plt.legend()
    plt.show()
    plt.close()

def read_data(folder, data_type):
    if data_type == "american_put":
        return read_american_put_data(folder)
    elif data_type == "variance_swap":
        return read_variance_swap_data(folder)


def read_american_put_data(folder, data_type, N_label=1):
    # read input parameters
    params = [[] for i in range(8)]  # K, r + 6 SVI parameters
    params_tex = [r"$K$", r"$r$", r"$a'$", r"$b$", r"$\rho$", r"$m$", r"$\sigma$", r"$\lambda$"]
    params_name = ["K", "r", "a1", "b", "rho", "m", "sigma", "lam"]

    targets = [[] for i in range(4)]  # price, delta, gamma, theta
    targets_tex = [r"$V$", r"$\Delta$", r"$\Gamma$", r"$\Theta$"]
    targets_name = ["price", "delta", "gamma", "theta"]

    for n in range(N_label):
        filename = f"{folder}/american_put_{data_type}_data_{n}.csv"
        if not os.path.exists(filename):
            print(f"File {filename} does not exist. Skipping...")
            continue
        data = np.loadtxt(filename, skiprows=1, delimiter=",")
        # Skip if the file doesn't exist

        if len(params[0]) == 0:
            for i in range(len(params)):
                params[i] = data[:, i]
            for i in range(len(targets)):
                targets[i] = data[:, i + len(params)]
        else:
            for i in range(len(params)):
                params[i] = np.concatenate((params[i], data[:, i]))
            for i in range(len(targets)):
                targets[i] = np.concatenate((targets[i], data[:, i + len(params)]))

    data_all = [np.array(params).T, params_tex, params_name, np.array(targets).T, targets_tex, targets_name]
    return data_all


def read_variance_swap_data(folder, data_type):
    # read input parameters
    data = np.loadtxt(f"{folder}/variance_swap_{data_type}_data.csv", skiprows=1, delimiter=",")
    params = [data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]]
    params_tex = [r"$r$", r"$a_1$", r"$b$", r"$\rho$", r"$m$", r"$\sigma$"]
    params_name = ["r", "a1", "b", "rho", "m", "sigma"]
    # read target
    Kvar = data[:, 6]
    targets = [Kvar, Kvar]
    targets_tex = [r"$K_{var}$", r"$K_{var}_1$"]
    targets_name = ["Kvar", "Kvar_place_holder"]

    data_all = [np.array(params).T, params_tex, params_name, np.array(targets).T, targets_tex, targets_name]
    return data_all


def calc_nearest_neighbor_distance(SqV, C):

    # Step 1: Build a k-d tree for efficient neighbor search
    tree = cKDTree(SqV)  # Use only spatial coordinates (x, y, z)
    distances, indices = tree.query(SqV, k=2)  # Find nearest neighbors (k=2)

    # Step 2: Compute color differences
    color_differences = np.abs(C - C[indices[:, 1]])

    # Step 3: Normalize by color range
    color_min = np.min(C)  # Minimum color value
    color_max = np.max(C)  # Maximum color value
    color_range = color_max - color_min  # Range of color values

    # Avoid division by zero if all color values are the same
    if color_range == 0:
        normalized_differences = np.zeros_like(color_differences)
    else:
        normalized_differences = color_differences / color_range * 2

    # Step 4: Compute average normalized color difference
    avg_normalized_difference = np.mean(normalized_differences)

    # print("Average Normalized Color Difference (by range):", avg_normalized_difference)
    return avg_normalized_difference


def targe_distribution(folder, data):
    params, params_tex, params_name, targets, targets_tex, targets_name = data

    # Extract each parameter for clarity
    K = params[:, 0]
    r = params[:, 1]
    sigma = params[:, 2]
    # Create a figure with 2x2 subplots for each target metric
    fig = plt.figure(figsize=(15, 12))
    # Loop over each target (price and Greeks)
    for i in range(targets.shape[1]):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        sc = ax.scatter(K, r, sigma, c=targets[:, i], cmap="viridis")

        # Set axis labels using TeX-formatted strings for clarity
        ax.set_xlabel(params_tex[0], fontsize=12)
        ax.set_ylabel(params_tex[1], fontsize=12)
        ax.set_zlabel(params_tex[2], fontsize=12)
        ax.set_title(f"{targets_name[i].capitalize()} ({targets_tex[i]})", fontsize=14)
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label(targets_tex[i], fontsize=12)

    plt.suptitle("Distribution of American Put Price and Greeks in Parameter Space (K, r, σ)", fontsize=16)
    plt.tight_layout(pad=0.1)

    # Save the figure to the specified folder
    file_path = folder.rstrip("/") + "/target_distribution.png"
    plt.savefig(file_path)
    plt.show()
    print(f"Target distribution plot saved to: {file_path}")


def auto_kernel(theta_init: dict, bounds: dict):
    """
    Build an RBF (+ White) kernel from two possible hyperparams:
      - 'length_scale'
      - 'noise_level'
    """
    kernel = None
    for name, init_val in theta_init.items():
        lb, ub = bounds[name]
        if name.lower() == "length_scale":
            comp = RBF(length_scale=init_val, length_scale_bounds=(lb, ub))
        elif name.lower() == "noise_level":
            comp = WhiteKernel(noise_level=init_val, noise_level_bounds=(lb, ub))
        else:
            raise ValueError(f"Unrecognized hyperparam name “{name}” – " "expected “length_scale” or “noise_level”")
        kernel = comp if kernel is None else (kernel + comp)
    return kernel


def GaussianProcess_optimization(folder, data_shuffled, perc_train, product_type):
    params, params_tex, params_name, targets, targets_tex, targets_name = data_shuffled
    # Unpack the input data
    params = params[: int(perc_train * len(params))]
    targets = targets[: int(perc_train * len(targets))]

    # Grid for hyperparameter search (for LML contour)
    grid_size = 20

    if product_type == "variance_swap":
        theta_per_target = {"Kvar": {"length_scale": np.linspace(1, 3, grid_size)}}
    elif product_type == "american_put":
        theta_per_target = {
            "price": {"length_scale": np.linspace(2.0, 3.0, grid_size), "noise_level": np.logspace(-4, -2, grid_size)},
            "delta": {"length_scale": np.linspace(1.0, 3.0, grid_size), "noise_level": np.logspace(-2, -1, grid_size)},
            "gamma": {"length_scale": np.linspace(1, 3, grid_size), "noise_level": np.logspace(-1, 0, grid_size)},
            "theta": {"length_scale": np.linspace(1.5, 3.0, grid_size), "noise_level": np.logspace(-3 , -1, grid_size)},
        }

    params_mean = np.mean(params, axis=0)
    params_std = np.std(params, axis=0)
    params_norm = (params - params_mean) / params_std

    # Normalize the targets (zero mean, unit variance)
    targets_mean = np.mean(targets, axis=0)
    targets_std = np.std(targets, axis=0)
    targets_norm = (targets - targets_mean) / targets_std

    # Create arrays with proper dimensions to ensure data alignment
    params_data = np.column_stack((params_name, params_mean, params_std))
    targets_data = np.column_stack((targets_name, targets_mean, targets_std))

    # Save the data as separate sections in the same file
    np.savetxt(f"{folder}/{product_type}_params_avg_std.txt", params_data, delimiter=",", header="params_name,params_mean,params_std", comments="", fmt="%s")
    np.savetxt(f"{folder}/{product_type}_targets_avg_std.txt", targets_data, delimiter=",", header="target_name,target_mean,target_std", comments="", fmt="%s")

    # Set up subplots for the LML contours per target.
    n_targets = len(targets_name)
    fig, axs = plt.subplots(1, n_targets, figsize=(6 * n_targets, 6), squeeze=True)
    axs = np.atleast_1d(axs)

    for idx, target_name in enumerate(targets_name):
        grid_spec = theta_per_target.get(target_name)
        if grid_spec is None:
            continue

        # build initial-center and bounds dicts
        init_dict = {p: grid_spec[p][len(grid_spec[p]) // 2] for p in grid_spec}
        bounds_dict = {p: (grid_spec[p][0], grid_spec[p][-1]) for p in grid_spec}

        # --- 3) fit a fixed GP to evaluate LML over the grid ---
        base_kernel = auto_kernel(init_dict, bounds_dict)
        gp0 = GaussianProcessRegressor(kernel=base_kernel, alpha=1e-6, optimizer=None)  # turn off built-in optimization
        gp0.fit(params_norm, targets_norm[:, idx])

        print("Training target:", target_name)

        # --- 4) compute LML grid (1D or 2D) ---
        param_names = list(grid_spec.keys())
        grids = [grid_spec[p] for p in param_names]
        mesh = list(product(*grids))
        shape = [g.size for g in grids]
        LML = np.zeros(shape)
        flat_LML = np.zeros(len(mesh))

        for i_pts, vals in enumerate(mesh):
            log_vals = np.log(vals)
            lml_val = gp0.log_marginal_likelihood(log_vals)/params_norm.shape[0]
            flat_LML[i_pts] = lml_val
            multi_idx = np.unravel_index(i_pts, shape)
            LML[multi_idx] = lml_val
            print(f"Evaluating LML at {vals}: {lml_val:.3f}, {i_pts}/{len(mesh)}")

        ax = axs[idx]
        if len(grids) == 1:
            ax.plot(grids[0], LML, label="LML")
            ax.set_xlabel(param_names[0])
        else:
            Xg, Yg = np.meshgrid(grids[0], grids[1], indexing="ij")
            cs = ax.contour(Xg, Yg, LML, levels=100)
            fig.colorbar(cs, ax=ax, label="LML")
            ax.set_yscale("log")
            ax.set_xlabel(param_names[0])
            ax.set_ylabel(param_names[1])

        # --- 5) full GP optimization ---
        if target_name == "gamma":
            # gamma is tooooo slow!
            gp_opt = GaussianProcessRegressor(kernel=auto_kernel(init_dict, bounds_dict), alpha=1e-5, n_restarts_optimizer=2)
        else:
            gp_opt = GaussianProcessRegressor(kernel=auto_kernel(init_dict, bounds_dict), alpha=1e-6, n_restarts_optimizer=5)
        gp_opt.fit(params_norm, targets_norm[:, idx])
        theta_opt = np.exp(gp_opt.kernel_.theta)
        lml_opt = gp_opt.log_marginal_likelihood(gp_opt.kernel_.theta)/params_norm.shape[0]

        # mark optimum
        if theta_opt.size == 1:
            ax.plot(theta_opt[0], lml_opt, "rx", markersize=10)
        else:
            ax.plot(theta_opt[0], theta_opt[1], "rx", markersize=10)
        ax.set_title(f"{target_name} opt: {theta_opt}")

        # --- 6) save everything into one .npz ---
        save_dict = {"LML": LML, "theta_opt": theta_opt, "LML_opt": lml_opt}
        # add each grid array
        for name, grid in zip(param_names, grids):
            save_dict[f"{name}_grid"] = grid

        np.savez_compressed(f"{folder}/{product_type}_{target_name}_LML.npz", **save_dict)

        # --- 7) pickle the optimized GP ---
        with open(f"{folder}/{product_type}_gp_{target_name}.pkl", "wb") as f:
            pickle.dump(gp_opt, f)

        print(f"Model and LML data for {product_type}: {target_name} saved.")

    # Save the average and standard deviation for the targets.

    plt.tight_layout()
    plt.savefig(f"{folder}/{product_type}_LML_plots.png", dpi=300)
    plt.show()
    plt.close()


def read_gp_and_params_stats(folder, data_shuffled, product_type):
    params, params_tex, params_name, target, target_tex, targets_name = data_shuffled
    params_stats = np.genfromtxt(f"{folder}/{product_type}_params_avg_std.txt", delimiter=",", skip_header=1, usecols=(1, 2))
    target_stats = np.genfromtxt(f"{folder}/{product_type}_targets_avg_std.txt", delimiter=",", skip_header=1, usecols=(1, 2))

    gp_per_params = {}
    for tname in targets_name:
        if os.path.exists(f"{folder}/{product_type}_gp_{tname}.pkl"):
            with open(f"{folder}/{product_type}_gp_{tname}.pkl", "rb") as f:
                gp_per_params[tname] = pickle.load(f)
    return params_stats, target_stats, gp_per_params


def GaussianProcess_prediction(folder, data, product_type):

    params, params_tex, params_name, target, target_tex, target_name = data
    params_stats, target_stats, gp_per_params = read_gp_and_params_stats(folder, data, product_type)
    params_mean, params_std = params_stats[:, 0], params_stats[:, 1]
    target_mean, target_std = target_stats[:, 0], target_stats[:, 1]
    # Unpack the input data
    params_test = params
    target_test = target

    # normalize the test data
    params_test = (params_test - params_mean) / params_std

    plt.figure()
    fig, axs = plt.subplots(1, len(target_name), figsize=(6 * len(target_name), 6))
    for tname, gp in gp_per_params.items():
        target_index = target_name.index(tname)
        Y = target_test[:, target_index]

        print("GPML kernel: %s" % gp.kernel_)
        gp_theta = np.exp(gp.kernel_.theta)  # gp.kernel_.theta return log transformed theta
        # kernel_params_array = np.array(list(kernel_params.values()))
        print("Kernel parameters:", gp_theta)
        print("Log-marginal-likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))

        start_time = time.perf_counter()
        Y_predict, Y_predict_err = gp.predict(params_test, return_std=True)
        elapsed_time = time.perf_counter() - start_time
        print(f"gp.predict time: {elapsed_time:.6f} seconds")
        # print("np.shape(test_data[:, 0])", np.shape(test_data[:, 0]))
        print("np.shape(Y_predict)", np.shape(Y_predict))

        # denormalize Y_predict
        Y_predict = Y_predict * target_std[target_index] + target_mean[target_index]
        Y_predict_err = Y_predict_err * target_std[target_index]

        axs[target_index].scatter(Y, Y_predict, marker="o", s=5, facecolor="none", edgecolor="black", label="data")
        axs[target_index].plot(Y, Y, "--")
        min_val = min(np.min(Y), np.min(Y_predict - Y_predict_err))
        max_val = max(np.max(Y), np.max(Y_predict + Y_predict_err))
        axs[target_index].set_xlim(min_val, max_val)
        axs[target_index].set_ylim(min_val, max_val)
        axs[target_index].set_xlabel(f"{target_tex[target_index]}")
        axs[target_index].set_ylabel(f"{target_tex[target_index]} Prediction")

        # save data to file
        data = np.column_stack((Y, Y_predict, Y_predict_err))
        column_names = [tname, "ML predicted", "ML predicted uncertainty"]
        np.savetxt(f"{folder}/{product_type}_{tname}_prediction.txt", data, delimiter=",", header=",".join(column_names), comments="")

    plt.savefig(f"{folder}/{product_type}_prediction.png", dpi=300)
    plt.close()
