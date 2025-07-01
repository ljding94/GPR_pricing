from SVI_vol import *
from American_put import *
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
from variance_swap import *


def plot_price_surface(svi_params, r, K, T):
    a1, b, rho, m, sigma, lam = svi_params
    S0 = 1
    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V = price_american_option_PSOR(
        S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="call", M=400, N=400, refine_theta=False
    )

    fig = plt.figure(figsize=(8, 15))
    ax = plt.subplot(211, projection="3d")
    ax2 = plt.subplot(212, projection="3d")
    # Create meshgrid for 3D plot
    N = 400
    S_grid, T_grid = np.meshgrid(S, np.linspace(0, T, N + 1))
    # Plot the 3D surface
    surf = ax.plot_surface(S_grid, T_grid, all_V.T, cmap="rainbow", linewidth=0, antialiased=True)
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=r"$V$")
    # Set labels and title
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$t$")
    ax.set_zlabel(r"$V$")
    ax.set_title("PSOR method")
    # plot the surface using CN method
    price, delta, gamma, theta, price_t0, delta_t0, gamma_t0, theta_t0, S_grid, V_all = price_american_option_CN(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="call", M=400, N=400)
    # Plot the 3D surface
    surf2 = ax2.plot_surface(S_grid, T_grid, V_all, cmap="rainbow", linewidth=0, antialiased=True)
    # Add a color bar
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, label=r"$V$")
    # Set labels and title
    ax2.set_xlabel(r"$S$")
    ax2.set_ylabel(r"$t$")
    ax2.set_zlabel(r"$V$")
    ax2.set_title("CN method")
    # Set title for the first subplot

    # Adjust view angle for better visualization
    plt.tight_layout()
    plt.savefig("../data/data_test/price_surface.png", dpi=300)
    plt.show()
    plt.close()


def plot_option_prices(S_grid, all_put_price, all_call_price, r, bs_vol):
    plt.plot(S_grid, all_put_price, label="American Put Price")
    plt.plot(S_grid, all_call_price, label="American Call Price")
    K = 1.0
    T = 1
    bs_price_call = bs_price(S_grid, K, T, r, bs_vol, type="call")
    bs_price_put = bs_price(S_grid, K, T, r, bs_vol, type="put")
    plt.plot(S_grid, bs_price_call, label="BS Call Price", linestyle="--")
    plt.plot(S_grid, bs_price_put, label="BS Put Price", linestyle="--")
    plt.xlabel(r"$S")
    plt.ylabel(r"$V$")
    plt.xlim(0.8, 1.2)
    plt.ylim(0, 0.2)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../data/data_test/option_prices.png", dpi=300)
    plt.show()


def plot_local_vol():
    r = 0.05
    a1, b, rho, m, sigma, lam = 0.03, 0.01, -0.4, 0.4, 0.8, 0.1
    S0 = 1.0
    Ss = np.linspace(0.5, 1.5, 100)
    ts = np.linspace(0.01, 1, 100)
    local_vols = np.zeros((len(Ss), len(ts)))
    for i, S in enumerate(Ss):
        for j, t in enumerate(ts):
            local_vols[i, j] = local_vol2(S0, S, t, r, a1, b, rho, m, sigma, lam)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Create meshgrid for 3D plot
    S_grid, T_grid = np.meshgrid(Ss, ts)

    # Plot the 3D surface
    surf = ax.plot_surface(S_grid, T_grid, local_vols.T, cmap="rainbow", linewidth=0, antialiased=True)

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=r"$\sigma^2_{loc}$")

    # Set labels and title
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$t$")
    ax.set_zlabel(r"$\sigma^2_{loc}$")

    # Adjust view angle for better visualization
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig("../data/data_test/local_vol_surface.png", dpi=300)
    plt.show()
    return local_vols


def plot_price_and_greeks(svi_params, r, K, T):
    a1, b, rho, m, sigma, lam = svi_params
    S0 = 1
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    # american call
    start_time = time.time()
    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V = price_american_option_PSOR(
        S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="call", M=500, N=500, refine_theta=False
    )
    print("PSOR calculation time:", time.time() - start_time)
    print("done calculation")
    print("np.max(S)", np.max(S))
    print("np.min(S)", np.min(S))
    print("len(S)", len(S))
    ni = 0
    nf = -1
    ax1.plot(S[ni:nf], price_grid[ni:nf], label="American Call (PSOR)")
    ax2.plot(S[ni:nf], delta_grid[ni:nf], label="American Call (PSOR)")
    ax3.plot(S[ni:nf], gamma_grid[ni:nf], label="American Call (PSOR)")
    ax4.plot(S[ni:nf], theta_grid[ni:nf], label="American Call (PSOR)")
    print("American Call Price:", price)

    start_time = time.time()
    price, delta, gamma, theta, price_t0, delta_t0, gamma_t0, theta_t0, S_grid, V_all = price_american_option_CN(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="call", M=500, N=500)
    print("CN calculation time:", time.time() - start_time)
    ax1.plot(S_grid, price_t0, label="American Call (CN)")
    ax2.plot(S_grid, delta_t0, label="American Call (CN)")
    ax3.plot(S_grid, gamma_t0, label="American Call (CN)")
    ax4.plot(S_grid, theta_t0, label="American Call (CN)")
    print("American Call Price (CN):", price)

    # american put
    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V = price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put", refine_theta=False)
    ax1.plot(S[ni:nf], price_grid[ni:nf], label="American Put (PSOR)")
    ax2.plot(S[ni:nf], delta_grid[ni:nf], label="American Put (PSOR)")
    ax3.plot(S[ni:nf], gamma_grid[ni:nf], label="American Put (PSOR)")
    ax4.plot(S[ni:nf], theta_grid[ni:nf], label="American Put (PSOR)")
    print("American Put Price:", price)

    price, delta, gamma, theta, price_t0, delta_t0, gamma_t0, theta_t0, S_grid, V_all = price_american_option_CN(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put", M=1000, N=1000)
    ax1.plot(S_grid, price_t0, label="American put (CN)")
    ax2.plot(S_grid, delta_t0, label="American put (CN)")
    ax3.plot(S_grid, gamma_t0, label="American put (CN)")
    ax4.plot(S_grid, theta_t0, label="American put (CN)")
    print("American Put Price (CN):", price)

    delta_grid, gamma_grid, S = calc_greek_by_bump(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put", M=1000, N=1000)
    ax2.plot(S, delta_grid, label="American Put (CN bump)")
    ax3.plot(S, gamma_grid, label="American Put (CN bump)")

    """
    ax1.set_xlim(0.8, 1.2)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0.8, 1.2)
    ax2.set_ylim(-1, 1)
    ax3.set_xlim(0.8, 1.2)
    ax3.set_ylim(-0.5, 4)
    ax4.set_xlim(0.8, 1.2)
    ax4.set_ylim(-0.5, 0.5)
    """

    # european call
    # European BS prices
    S_bs = S[ni:nf]
    bs_price_call, bs_delta_call, bs_gamma_call, bs_theta_call = bs_price(S_bs, K, T, r, np.sqrt(a1), type="call")
    ax1.plot(S_bs, bs_price_call, label="European Call (BS)", linestyle="--", color="black")
    ax2.plot(S_bs, bs_delta_call, label="European Call (BS)", linestyle="--", color="black")
    ax3.plot(S_bs, bs_gamma_call, label="European Call (BS)", linestyle="--", color="black")
    ax4.plot(S_bs, bs_theta_call, label="European Call (BS)", linestyle="--", color="black")
    bs_price_put, bs_delta_put, bs_gamma_put, bs_theta_put = bs_price(S_bs, K, T, r, np.sqrt(a1), type="put")
    ax1.plot(S_bs, bs_price_put, label="European Put (BS)", linestyle=":", color="black")
    ax2.plot(S_bs, bs_delta_put, label="European Put (BS)", linestyle=":", color="black")
    ax3.plot(S_bs, bs_gamma_put, label="European Put (BS)", linestyle=":", color="black")
    ax4.plot(S_bs, bs_theta_put, label="European Put (BS)", linestyle=":", color="black")

    # TODO: implement up and down bump to calculate the greeks

    """
    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S = price_american_option_PSOR_mc(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="call")
    ax1.plot(S, price_grid, label="American Call (PSOR_mc)")
    ax2.plot(S, delta_grid, label="American Call (PSOR_mc)")
    ax3.plot(S, gamma_grid, label="American Call (PSOR_mc)")
    ax4.plot(S, theta_grid, label="American Call (PSOR_mc)")
    print("American Call Price (MC):", price)


    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S = price_american_option_PSOR_mc(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put")
    ax1.plot(S, price_grid, label="American Put (PSOR_mc)")
    ax2.plot(S, delta_grid, label="American Put (PSOR_mc)")
    ax3.plot(S, gamma_grid, label="American Put (PSOR_mc)")
    ax4.plot(S, theta_grid, label="American Put (PSOR_mc)")
    print("American Put Price (MC):", price)
    """

    ax1.set_xlabel(r"$S$")
    ax1.set_ylabel(r"$V$")
    ax1.set_title("Price")
    ax2.set_xlabel(r"$S$")
    ax2.set_ylabel(r"$\Delta$")
    ax2.set_title("Delta")
    ax3.set_xlabel(r"$S$")
    ax3.set_ylabel(r"$\Gamma$")
    ax3.set_title("Gamma")
    ax4.set_xlabel(r"$S$")
    ax4.set_ylabel(r"$\Theta$")
    ax4.set_title("Theta")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.tight_layout()
    plt.savefig("../data/data_test/price_and_greeks.png", dpi=300)
    plt.show()
    plt.close()


def test_line_overlap():
    x = np.linspace(0.5, 1.5, 100)
    y = np.sin(np.pi * x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linestyle=(0, (1, 4)), label="Line 0")
    plt.plot(x, y + 0.1, linestyle=(0, (2, 3)), label="Line 1")
    plt.plot(x, y + 0.2, linestyle=(-1, (2, 3)), label="Line 2")
    plt.plot(x, y + 0.3, linestyle=(-2, (2, 3)), label="Line 3")
    plt.plot(x, y + 0.4, linestyle=(-3, (2, 3)), label="Line 4")
    plt.plot(x, y + 0.5, linestyle=(-4, (2, 3)), label="Line 5")
    plt.legend
    plt.show()


def test_variance_swap_running_time():
    folder = "../data/data_test"
    start = time.time()
    elapsed = 0
    generate_variance_swap_data(folder, 2000, data_type="test")
    elapsed = time.time() - start
    print(f"Generated in {elapsed:.4f} seconds")


def test_american_put_running_time():
    folder = "../data/data_test"
    # M,N= 1000, 1000
    # M,N= 500, 500
    M, N = 200, 200
    n_prices = 0
    start = time.time()
    elapsed = 0
    while elapsed < 300:
        generate_american_put_data_set(folder, f"{M}x{N}", 1, data_type="test", M=M, N=N)
        n_prices += 1
        elapsed = time.time() - start
        print(f"Generated {n_prices} prices for M={M}, N={N} in {elapsed:.4f} seconds")


def test_atm_vol_range():

    N_data = 100000
    S0 = 1

    T = 1.0
    # logK_vals = np.random.uniform(-0.15, 0.15, N_data)  # log strike prices
    K_vals = np.random.uniform(0.85, 1.15, N_data)  # strike prices
    # K_vals = S0 * np.exp(logK_vals)  # strike prices
    r_vals = np.random.uniform(0.00, 0.06, N_data)  # risk-free rate
    # SVI parameters
    a1_vals = np.random.uniform(0.00, 0.02, N_data)  # a1>0
    b_vals = np.random.uniform(0.00, 0.3, N_data)  # b>0
    rho_vals = np.random.uniform(-0.4, 0.8, N_data)  # |rho|<1
    m_vals = np.random.uniform(-0.2, 0.6, N_data)  # m unlimited
    sigma_vals = np.random.uniform(0.00, 1.0, N_data)  # sigma>0
    lam_vals = np.random.uniform(0.00, 1.0, N_data)

    w0 = calc_raw_SVI_skew_T1(0, a1_vals, b_vals, rho_vals, m_vals, sigma_vals)
    bs_vol = np.sqrt(w0)
    print("min(bs_vol), max(bs_vol)")
    print(min(bs_vol), max(bs_vol))
    plt.figure()
    plt.hist(bs_vol, bins=100, density=True, alpha=0.5, color="blue", label="BS Vol")
    plt.xlabel("ATM Volatility")
    plt.ylabel("Density")
    plt.title("Distribution of ATM Volatility")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()


def test_atm_vol():
    w0 = calc_raw_SVI_skew_T1(0, 0.01, 0.15, -0.1, 0.2, 0.2)
    bs_vol = np.sqrt(w0)
    print("ATM Variance:", w0)
    print("ATM Volatility:", bs_vol)



def main():
    test_atm_vol()
    #test_atm_vol_range()

    # test_variance_swap_running_time()
    # test_american_put_running_time()
    return 0
    # S, P, lbd = build_binomial_tree(100, 5, 0.03, 5, (100, 0.1, -0.05), "Derman")
    # S, P, lbd = build_binomial_tree(100, 2, 0.03, 2, (0.1,0), "flat")
    # test_line_overlap()
    # return 0
    # res = flat_vol_american_put_tree_greeks(100, 100, 0.03, 0.1, 1, 100)
    # print(res)
    S0 = 1.0
    r = 0.00
    T = 1
    K = 1.0

    # plot_local_vol()

    # a1, b, rho, m, sigma, lam = 0.03, 0.01, -0.4, 0.2, 0.3, 0.1
    a1, b, rho, m, sigma, lam = 0.03, 0.02, -0.4, 0.2, 0.3, 0.1
    a1, b, rho, m, sigma, lam = 0.03, 0.0, 0, 0.0, 0.01, 0.0
    # plot_price_surface((a1, b, rho, m, sigma, lam), r, K, T)
    plot_price_and_greeks((a1, b, rho, m, sigma, lam), r, K, T)
    return 0

    # put_price, all_put_price, S_grid = price_american_option_SVI(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put")
    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S = price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put")
    print("put price and greeks")

    # call_price, all_call_price, S_grid = price_american_option_SVI(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type='call')
    price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S = price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="call")

    F = S0 * np.exp(r * T)
    bs_vol2 = calc_raw_SVI_surface(np.log(K / F), T, a1, b, rho, m, sigma, lam) * T

    # plot_option_prices(S_grid, all_put_price, all_call_price, r, np.sqrt(bs_vol2))

    print(f"American Put Price: {put_price:.4f}")
    print(f"American Call Price: {call_price:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
