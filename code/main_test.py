from SVI_vol import *
from American_put import *
import time
import matplotlib.pyplot as plt


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


def main():
    # S, P, lbd = build_binomial_tree(100, 5, 0.03, 5, (100, 0.1, -0.05), "Derman")
    # S, P, lbd = build_binomial_tree(100, 2, 0.03, 2, (0.1,0), "flat")

    # res = flat_vol_american_put_tree_greeks(100, 100, 0.03, 0.1, 1, 100)
    # print(res)
    S0 = 1.0
    r = 0.05
    T = 1
    K = 0.9

    # plot_local_vol()

    a1, b, rho, m, sigma, lam = 0.03, 0.01, -0.4, 0.2, 0.3, 0.1
    # put_price, all_put_price, S_grid = price_american_option_SVI(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put")
    put_price, delta, gamma_val, theta, all_put_price, S_grid = price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put")
    print("put price and greeks")
    print(put_price, delta, gamma_val, theta)

    # call_price, all_call_price, S_grid = price_american_option_SVI(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type='call')
    call_price, delta, gamma_val, theta, all_call_price, S_grid = price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="call")

    F = S0 * np.exp(r * T)
    bs_vol2 = calc_raw_SVI_surface(np.log(K / F), T, a1, b, rho, m, sigma, lam) * T



    #plot_option_prices(S_grid, all_put_price, all_call_price, r, np.sqrt(bs_vol2))

    print(f"American Put Price: {put_price:.4f}")
    print(f"American Call Price: {call_price:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
