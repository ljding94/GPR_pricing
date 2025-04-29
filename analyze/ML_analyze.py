import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import os
from scipy.optimize import curve_fit
import pickle
from scipy.spatial import cKDTree


def read_data(folder, data_type):
    if data_type == "american_put":
        return read_american_put_data(folder)
    elif data_type == "variance_swap":
        return read_variance_swap_data(folder)


def read_american_put_data(folder):
    # read input parameters
    data = np.loadtxt(f"{folder}/american_put_data.csv", skiprows=1, delimiter=",")
    params = [data[:, 1], data[:, 2], data[:, 3]]
    params_tex = [r"$K$", r"$r$", r"$\sigma$"]
    params_name = ["K", "r", "sigma"]
    # read target
    price = data[:, 5]
    delta = data[:, 6]
    gamma = data[:, 7]
    theta = data[:, 8]
    targets = [price, delta, gamma, theta]
    targets_tex = [r"$P$", r"$\Delta$", r"$\Gamma$", r"$\Theta$"]
    targets_name = ["price", "delta", "gamma", "theta"]

    data_all = [np.array(params).T, params_tex, params_name, np.array(targets).T, targets_tex, targets_name]
    return data_all


def read_variance_swap_data(folder):
    # read input parameters
    data = np.loadtxt(f"{folder}/variance_swap_data.csv", skiprows=1, delimiter=",")
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


def GaussianProcess_optimization(folder, data_shuffled, perc_train, data_type):
    params, params_tex, params_name, targets, targets_tex, targets_name = data_shuffled
    # Unpack the input data
    params = params[: int(perc_train * len(params))]
    targets = targets[: int(perc_train * len(targets))]

    # Grid for hyperparameter search (for LML contour)
    grid_size = 10
    # Only optimizing over hyperparameters for the "price" target in this example;
    # extend to additional targets as needed.
#    theta_per_target = {
        # following are for the american put
        # "price": (np.logspace(-2, 1, grid_size), np.logspace(-6, -2, grid_size)),
        # "delta": (np.logspace(-1, 1, grid_size), np.logspace(-8, -3, grid_size)),
        #"gamma": (np.logspace(-1, 1, grid_size), np.logspace(-8, -3, grid_size)),
        #"theta": (np.logspace(-1, 1, grid_size), np.logspace(-8, -3, grid_size)),
    #}
    theta_per_target = {
        # following are for the american put
        "price": np.logspace(-2, 1, grid_size),
        # "delta": np.logspace(-1, 1, grid_size),
        #"gamma": np.logspace(-1, 1, grid_size),
        #"theta": np.logspace(-1, 1, grid_size),
        # for variance swap
        "Kvar": np.logspace(-2, 1, grid_size)
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
    np.savetxt(f"{folder}/{data_type}_params_avg_std.txt", params_data, delimiter=",", header="params_name,params_mean,params_std", comments="", fmt="%s")
    np.savetxt(f"{folder}/{data_type}_targets_avg_std.txt", targets_data, delimiter=",", header="target_name,target_mean,target_std", comments="", fmt="%s")

    gp_per_target = {}

    # Set up subplots for the LML contours per target.
    n_targets = len(targets_name)
    fig, axs = plt.subplots(1, n_targets, figsize=(6 * n_targets, 6), squeeze=False)

    for idx, target_name in enumerate(targets_name):
        if target_name not in theta_per_target:
            # Skip targets that are not defined in our hyperparameter grid,
            # or add default hyperparameter ranges if desired.
            continue

        print("Training target:", target_name)
        target_index = targets_name.index(target_name)
        # Feature matrix remains the same for all targets (input: [K, r, sigma])
        F_learn = params_norm

        # ------------------------------
        # INITIAL GP FIT (no optimization)
        # ------------------------------
        # First, set up a kernel with initial hyperparameters and no optimizer.
        #kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        kernel = RBF(length_scale=1.0)
        # Fit GPR without optimizing the kernel parameters (initial fit)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
        gp.fit(F_learn, targets_norm[:, target_index])
        print("Initial GP kernel:", gp.kernel_)
        init_theta = np.exp(gp.kernel_.theta)  # Note: kernel_.theta is in log-space.
        print("Initial kernel parameters (theta):", init_theta)
        print("Initial Log Marginal Likelihood: %.3f" % gp.log_marginal_likelihood(gp.kernel_.theta))

        # -----------------------------------------------------------
        # COMPUTE LML ON A 1D GRID (for the only hyperparameter: length scale)
        # -----------------------------------------------------------
        theta_range = theta_per_target[target_name]
        LML = []
        # Evaluate LML for each candidate length scale (passing its log value)
        for theta_val in theta_range:
            lml_val = gp.log_marginal_likelihood(np.array([np.log(theta_val)]))
            LML.append(lml_val)

            print(f"Computing LML for {target_name}: {len(LML)}/{grid_size} completed")
            print(f"theta_val = {theta_val}, LML = {lml_val}")


        LML = np.array(LML)

        ax = axs[0, target_index]
        ax.plot(theta_range, LML, label="LML")

        # -----------------------------------------------------------
        # OPTIMIZATION OF THE GP HYPERPARAMETER (length scale only)
        # -----------------------------------------------------------
        # Use the midpoint of theta_range as an initial guess.
        init_theta_val = theta_range[grid_size // 2]
        kernel = RBF(length_scale=init_theta_val, length_scale_bounds=(theta_range[0], theta_range[-1]))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)
        gp.fit(F_learn, targets_norm[:, target_index])
        gp_per_target[target_name] = gp

        print("\nOptimized GP kernel for {}: {}".format(target_name, gp.kernel_))
        opt_theta = np.exp(gp.kernel_.theta)  # optimized length scale
        print("Optimized kernel length scale (theta):", opt_theta)
        opt_lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        print("Optimized Log Marginal Likelihood: %.3f" % opt_lml)

        # Plot the optimized hyperparameter on the LML curve.
        #print(len(opt_theta), len(opt_lml))
        print("opt_theta, opt_lml")
        print(opt_theta, opt_lml)
        ax.plot(opt_theta, [opt_lml], "x", color="red", markersize=10, markeredgewidth=2, label=f"l={opt_theta[0]:.2e}")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\ell$ (length scale)")
        ax.set_ylabel("Log Marginal Likelihood")
        ax.set_title(f"LML vs. Length Scale for {target_name}")
        ax.legend()

        # ------------------------------
        # SAVE LML GRID DATA AND MODEL
        # ------------------------------
        data_save = np.column_stack((np.full(len(theta_range), opt_theta), theta_range, LML))
        column_names = ["gp_theta", "theta", "LML"]
        np.savetxt(f"{folder}/{data_type}_{target_name}_LML.txt", data_save, delimiter=",",
                   header=",".join(column_names), comments="", fmt="%.6e")
        with open(f"{folder}/{data_type}_gp_{target_name}.pkl", "wb") as f:
            pickle.dump(gp, f)
        print(f"Model and LML data for {data_type}: {target_name} saved.")

    # Save the average and standard deviation for the targets.

    plt.tight_layout()
    plt.savefig(f"{folder}/{data_type}_LML_subplots.png", dpi=300)
    plt.show()
    plt.close()


def read_gp_and_params_stats(folder, data_shuffled, data_type):
    params, params_tex, params_name, target, target_tex, target_name = data_shuffled
    params_stats = np.genfromtxt(f"{folder}/{data_type}_params_avg_std.txt", delimiter=",", skip_header=1, usecols=(1, 2))
    target_stats = np.genfromtxt(f"{folder}/{data_type}_targets_avg_std.txt", delimiter=",", skip_header=1, usecols=(1, 2))

    gp_per_params = {}
    for tname in target_name:
        if os.path.exists(f"{folder}/{data_type}_gp_{tname}.pkl"):
            with open(f"{folder}/{data_type}_gp_{tname}.pkl", "rb") as f:
                gp_per_params[tname] = pickle.load(f)
    return params_stats, target_stats, gp_per_params


def GaussianProcess_prediction(folder, data_shuffled, perc_train, data_type):

    params, params_tex, params_name, target, target_tex, target_name = data_shuffled
    params_stats, target_stats, gp_per_params = read_gp_and_params_stats(folder, data_shuffled, data_type)
    params_mean, params_std = params_stats[:, 0], params_stats[:, 1]
    target_mean, target_std = target_stats[:, 0], target_stats[:, 1]
    # Unpack the input data
    params_test = params[int(perc_train * len(params)) :]
    target_test = target[int(perc_train * len(target)) :]

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

        Y_predict, Y_predict_err = gp.predict(params_test, return_std=True)
        # print("np.shape(test_data[:, 0])", np.shape(test_data[:, 0]))
        print("np.shape(Y_predict)", np.shape(Y_predict))

        # denormalize Y_predict
        Y_predict = Y_predict * target_std[target_index] + target_mean[target_index]
        Y_predict_err = Y_predict_err * target_std[target_index]

        axs[target_index].scatter(Y, Y_predict, marker="o")
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
        np.savetxt(f"{folder}/{data_type}_{tname}_prediction.txt", data, delimiter=",", header=",".join(column_names), comments="")

    plt.savefig(f"{folder}/{data_type}_prediction.png", dpi=300)
    plt.close()


def ax_fit(x, a):
    return a * x


def fit_Rg2(q, Sq):
    popt, pcov = curve_fit(ax_fit, q**2 / 3, (1 - Sq))
    perr = np.sqrt(np.diag(pcov))
    return popt[0], perr[0]


def calc_Sq_fitted_Rg2(folder, parameters_test, params_names):
    segment_type, params, params_names, Sq, Sq_err, q = read_Sq_data(folder, parameters_test)

    MC_Rg2 = params[:, params_names.index("Rg2")]
    # qfns = [10,20,30,40]
    qfns = [50, 55, 60, 65, 70]
    Rg2s = []
    Rg2_errs = []
    plt.figure()
    for qfn in qfns:
        Rg2s.append([])
        Rg2_errs.append([])
        for i in range(len(Sq)):
            Rg2, Rg2_err = fit_Rg2(q[:qfn], Sq[i][:qfn])
            Rg2s[-1].append(Rg2)
            Rg2_errs[-1].append(Rg2_err)

        plt.scatter(MC_Rg2, Rg2s[-1], alpha=0.5, label=f"qf={q[qfn-1]}")
    plt.plot(MC_Rg2, MC_Rg2, "k--")
    plt.xlabel("MC Rg2")
    plt.ylabel("Fitted Rg2")
    plt.legend()
    plt.savefig(f"{folder}/{segment_type}_Rg2_fit.png", dpi=300)
    plt.close()

    data = np.column_stack(([MC_Rg2] + Rg2s))
    column_names = ["MC Rg2", "fitted Rg2"]
    np.savetxt(f"{folder}/data_{segment_type}_fitted_Rg2.txt", data, delimiter=",", header=",".join(column_names), comments="")
