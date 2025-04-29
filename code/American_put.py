import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from SVI_vol import *
from scipy.linalg import solve_banded
import QuantLib as ql
import time


def flat_vol_american_put_tree_greeks(S, K, r, sigma, T, N):
    """
    Price an American put option using a full binomial tree and compute some Greeks
    directly from the tree nodes. This method constructs the full tree of stock prices
    and option values, from which the first-step approximations for Delta, Gamma, and Theta
    are derived.

    Parameters:
        S (float): Current underlying asset price.
        K (float): Strike price.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        T (float): Time to maturity.
        N (int): Number of time steps.

    Returns:
        tuple: (option_price, Delta, Gamma, Theta)
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))  # up factor
    d = 1 / u  # down factor
    p = (np.exp(r * dt) - d) / (u - d)  # risk-neutral probability

    # Initialize arrays to store stock prices and option values.
    stock = np.zeros((N + 1, N + 1))
    option = np.zeros((N + 1, N + 1))

    # Fill in stock prices for each node.
    for i in range(N + 1):
        for j in range(i + 1):
            stock[i, j] = S * (u**j) * (d ** (i - j))

    # Compute option values at maturity (time T).
    option[N, :] = np.maximum(K - stock[N, :], 0)

    # Backward induction to fill in option values at earlier times.
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation = np.exp(-r * dt) * (p * option[i + 1, j + 1] + (1 - p) * option[i + 1, j])
            exercise = K - stock[i, j]
            option[i, j] = max(continuation, exercise)

    # The option price is at the root of the tree.
    option_price = option[0, 0]
    print(f"Option price: {option_price}")

    # --- Direct Greek calculations from the tree ---
    # Delta computed from the two nodes at the first step (i = 1)
    delta = (option[1, 1] - option[1, 0]) / (stock[1, 1] - stock[1, 0])

    # Gamma computed using nodes at the second time step (i = 2)
    # Calculate delta at the "up" branch at time 2:
    delta_up = (option[2, 2] - option[2, 1]) / (stock[2, 2] - stock[2, 1])
    # Calculate delta at the "down" branch at time 2:
    delta_down = (option[2, 1] - option[2, 0]) / (stock[2, 1] - stock[2, 0])
    # Then, approximate Gamma by the difference between these two deltas.
    gamma = (delta_up - delta_down) / ((stock[2, 2] - stock[2, 0]) / 2)

    # Theta can be estimated using the difference between the value at time 0 and one step ahead.
    # One common approximation uses the middle node at time 1:
    theta = (option[1, 1] - option[0, 0]) / dt

    return option_price, delta, gamma, theta


def flat_vol_generate_american_put_data_set(folder):
    """
    Generates a data set of American put option prices and Greeks for a range of strike prices
    and volatilities, and saves the results to a CSV file.

    Parameters:
        folder (str): The folder (path) where the CSV file will be saved.
    """
    S = 1.0
    T = 1.0
    r = 0.05
    rs = np.linspace(0.00, 0.06, 20)
    Ks = np.linspace(0.8, 1.2, 20)
    sigmas = np.linspace(0.1, 0.5, 20)
    N = 220

    # List to hold each data row.
    data_rows = []

    # Generate parameter set based on K and sigma.
    total = len(Ks) * len(sigmas) * len(rs)
    count = 0

    for K in Ks:
        for sigma in sigmas:
            for r in rs:
                price, delta, gamma, theta = flat_vol_american_put_tree_greeks(S, K, r, sigma, T, N)
                # Create a row containing the parameters and computed metrics.
                row = [S, K, r, sigma, T, price, delta, gamma, theta]
                data_rows.append(row)

                # Update progress
                count += 1
                if count % 100 == 0 or count == total:
                    print(f"Progress: {count}/{total} ({count/total:.1%})")

    data = np.array(data_rows)

    # Create a proper file path.
    file_path = f"{folder}/american_put_data.csv"

    # Save data to a CSV file.
    header = "S,K,r,sigma,T,price,delta,gamma,theta"
    np.savetxt(file_path, data, delimiter=",", header=header, comments="")
    print(f"Data saved to: {file_path}")


def american_put_vol_binomial(K, T, S0, r, svi_params, N):
    a1, b, rho, m, sigma, lam = svi_params
    dt = T / N

    # Set the log-price step size: small constant (controls the accuracy)
    dx = 0.1

    # Build grid of log-prices centered at x0 = log(S0)
    x0 = np.log(S0)
    grid = [x0 + (j - N) * dx for j in range(2 * N + 1)]  # full grid of log-prices

    # Option values at each time step (each is a 1D array indexed by log-price offset)
    V = [np.zeros(2 * N + 1) for _ in range(N + 1)]
    S_vals = np.exp(grid)

    # Terminal payoff at maturity
    for j in range(2 * N + 1):
        V[N][j] = max(K - S_vals[j], 0)

    # Backward induction
    for i in reversed(range(N)):
        t = i * dt
        for j in range(N - i, N + 1 + i):  # nodes reachable at time i
            x = grid[j]
            S = np.exp(x)
            k = np.log(S / S0)
            T_rem = T - t
            sigma_loc_sq = local_var(k, T_rem, a1, b, rho, m, sigma, lam)

            # Risk-neutral up probability
            mu = r - 0.5 * sigma_loc_sq
            p = 0.5 + 0.5 * (mu * dt / dx)

            # Find child indices
            j_up = j + 1
            j_dn = j - 1

            # Discounted expected value
            cont_val = np.exp(-r * dt) * (p * V[i + 1][j_up] + (1 - p) * V[i + 1][j_dn])
            V[i][j] = max(K - S, cont_val)  # early exercise value

    # Return option price at the root node (log S = log S0)
    j0 = grid.index(x0)
    return V[0][j0]


def ql_american_option_price(K, T, S0, r, svi_params, N, option_type="put"):
    """
    Price an American option (put or call) using QuantLib and SVI-implied vol surface.

    Parameters:
    - K: strike price
    - T: time to maturity (in years)
    - S0: spot price
    - r: risk-free interest rate (annualized)
    - svi_params: tuple (a1, b, rho, m, sigma, lam)
    - option_type: 'put' or 'call'

    Returns:
    - American option price
    """
    a1, b, rho, m, sigma, lam = svi_params

    def svi_vol(k, T):
        total_var = calc_raw_SVI_surface(k, T, a1, b, rho, m, sigma, lam)
        return np.sqrt(total_var / T)

    # Setup QuantLib environment
    settlement = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = settlement
    day_count = ql.Actual365Fixed()

    # Build N expiry dates uniformly between today and T
    maturity_times = np.linspace(0.01, T, N)  # start slightly > 0 to avoid T=0
    expiry_dates = [settlement + ql.Period(int(t * 365), ql.Days) for t in maturity_times]

    # Build N strikes around S0
    strikes = np.linspace(0.5 * S0, 1.5 * S0, N)

    # Vol matrix: vol[time][strike]
    vol_matrix = []
    for t, expiry in zip(maturity_times, expiry_dates):
        vols_at_T = []
        for K_i in strikes:
            k = np.log(K_i / S0)
            vols_at_T.append(svi_vol(k, t))
        vol_matrix.append(vols_at_T)

    # Build Black variance surface
    calendar = ql.NullCalendar()
    black_surface = ql.BlackVarianceSurface(settlement, calendar, expiry_dates, list(strikes), vol_matrix, day_count)

    # Set up process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(settlement, r, day_count))
    dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(settlement, 0.0, day_count))

    bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_handle, rate_handle, ql.BlackVolTermStructureHandle(black_surface))

    # Define American Option
    if option_type.lower() == "put":
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    elif option_type.lower() == "call":
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    else:
        raise ValueError("option_type must be 'put' or 'call' ")

    final_expiry = settlement + ql.Period(int(T * 365), ql.Days)
    exercise = ql.AmericanExercise(settlement, final_expiry)

    option = ql.VanillaOption(payoff, exercise)

    # Use Finite Difference engine
    engine = ql.FdBlackScholesVanillaEngine(bsm_process, timeSteps=200, gridPoints=200)
    option.setPricingEngine(engine)

    return option.NPV()


# ---- American Option Pricing Solver ----
def price_american_option_SVI(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put", S_max_mult=10.0, M=2000, N=2000):
    """
    Price an American option (put or call) using SVI-derived local volatility and finite difference method.

    Parameters:
    - S0: initial stock price
    - K: strike price
    - r: risk-free rate
    - T: maturity
    - a1, b, rho, m, sigma, lam: SVI parameters
    - option_type: 'put' or 'call'
    - S_max_mult: factor for maximum stock price in grid
    - M: number of stock grid points
    - N: number of time steps

    Returns:
    - price: American option price
    """

    # 1. Set up grid
    S_max = S_max_mult * K
    dS = (S_max - 0) / M

    # Ensure S0 is exactly on the grid
    i_S0 = round(S0 / dS)
    S0_exact = i_S0 * dS

    S_grid = np.linspace(0, S_max, M + 1)

    dt = T / N

    if option_type == "put":
        payoff = np.maximum(K - S_grid, 0)
    elif option_type == "call":
        payoff = np.maximum(S_grid - K, 0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")

    V = payoff.copy()

    # 2. Prepare coefficients
    alpha = np.zeros(M + 1)
    beta = np.zeros(M + 1)
    gamma = np.zeros(M + 1)

    # 3. Time-stepping backward
    for n in range(N - 1, -1, -1):
        t = n * dt

        # Update forward price at current time t
        F_t = S0 * np.exp(r * t)
        # choose dt_n and θ for Rannacher smoothing
        if n >= N - 2:
            dt_n = dt / 2
            theta = 1.0  # backward Euler
        else:
            dt_n = dt
            theta = 0.5  # Crank–Nicolson
        # Build tridiagonal matrix
        for i in range(1, M):
            # T_remain = T - t if T - t > 1e-8 else 1e-8  # Avoid division by zero
            # TODO: I'm not sure if this is correct, of
            # is sigma_loc^2 (S,t) = vL(k=log(S/ S0e^{rt}), T-t)?????????????

            # loc_var = local_var(np.log(S_grid[i] / F_t), T_remain, a1, b, rho, m, sigma, lam)

            loc_var = local_var(np.log(S_grid[i] / F_t), t, a1, b, rho, m, sigma, lam)

            vol2 = loc_var

            vol2 = 0.1**2  # testing

            Si = S_grid[i]
            alpha[i] = 0.5 * (vol2 * Si**2 / dS**2 - r * Si / dS)
            beta[i] = -(vol2 * Si**2 / dS**2 + r)
            gamma[i] = 0.5 * (vol2 * Si**2 / dS**2 + r * Si / dS)

        if option_type == "put":
            V[0], V[M] = K, 0.0
        else:
            V[0], V[M] = 0.0, S_max - K

        # Set up RHS vector
        RHS = V.copy()
        i_idx = np.arange(1, M)
        RHS[i_idx] += (1 - theta) * dt_n * (alpha[i_idx] * V[i_idx - 1] + beta[i_idx] * V[i_idx] + gamma[i_idx] * V[i_idx + 1])

        # implicit boundary injections: θ dt_n α₁ V₀ and θ dt_n γ_{M-1} V_M
        RHS[1] += theta * dt_n * alpha[1] * V[0]
        RHS[M - 1] += theta * dt_n * gamma[M - 1] * V[M]

        # build banded matrix with corrected bands
        ab = np.zeros((3, M - 1))
        ab[0, : M - 1] = -theta * dt_n * gamma[1:M]
        # main diag: 1 - θ dt_n * β_{i}
        ab[1, :] = 1.0 - theta * dt_n * beta[1:M]
        # lower diagonal: -θ dt_n * α_{i}, for i=2…M-1 → ab[2, j] with j=i-1
        ab[2, 1:] = -theta * dt_n * alpha[2:M]

        # solve and apply early‐exercise
        V_inner = solve_banded((1, 1), ab, RHS[1:M])
        V[1:M] = np.maximum(V_inner, payoff[1:M])
        # V[1:M] = np.maximum(V_inner, payoff[1:M])  # Early exercise condition
        V[1:M] = V_inner

    # Directly pick price at S0 grid point
    return float(V[i_S0]), V, S_grid
    # Interpolate to find price at S0
    # from scipy.interpolate import interp1d
    # price_func = interp1d(S_grid, V, kind='linear')
    # return float(price_func(S0))


def _psor_step(alpha, beta, gamma, b, payoff, option_type="put", omega=1.2, tol=1e-8, maxit=5000):
    """
    Solve M V = b with LCP constraint V >= payoff via PSOR,
    where M has tridiagonal entries:
      M[i,i-1] = -alpha[i],   M[i,i] =  beta[i],   M[i,i+1] = -gamma[i].
    """
    M_size = len(beta)
    V = payoff.copy()
    for it in range(maxit):
        err = 0.0
        # interior nodes only
        for i in range(1, M_size - 1):
            # residual r_i = b_i - sum_j M_ij V_j
            r_i = b[i] - (-alpha[i] * V[i - 1] + beta[i] * V[i] - gamma[i] * V[i + 1])
            update = V[i] + omega * (r_i / beta[i])
            if option_type == "put":
                V_new = max(payoff[i], update)
            else:
                V_new = update
            err = max(err, abs(V_new - V[i]))
            V[i] = V_new
        if err < tol:
            break
    return V


def make_grid_with_S0_on_node(S0, S_max, M):
    # 1) raw step
    dS = S_max / M

    # 2) index nearest to S0
    i0 = int(round(S0 / dS))

    # 3) lower bound so that S[i0] == S0
    S_min = S0 - i0 * dS

    # 4) build grid
    S = S_min + np.arange(M + 1) * dS
    return S, dS, i0


def price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put", S_max_mult=3.0, M=400, N=400, omega=1.2, tol=1e-8):  # SVI ignored here
    # 1. Spatial grid in S
    # usage inside your solver:
    S_max = S_max_mult * K
    S, dS, i_S0 = make_grid_with_S0_on_node(S0, S_max, M)

    # 2. Payoff and initial condition
    if option_type == "put":
        payoff = np.maximum(K - S, 0.0)
    else:
        payoff = np.maximum(S - K, 0.0)
    V = payoff.copy()
    V_one_step = None  # for calculating theta

    dt = T / N
    alpha = np.zeros(M + 1)
    beta = np.zeros(M + 1)
    gamma = np.zeros(M + 1)

    # time‐stepping backward
    for n in range(N - 1, -1, -1):
        t = n * dt
        # Rannacher smoothing
        if n >= N - 2:
            dt_n, theta = dt / 2, 1.0
        else:
            dt_n, theta = dt, 0.5

        # build A-coeffs at this layer (flat vol test; replace with local_var if you like)
        for i in range(1, M):
            Si = S[i]
            vol2 = local_vol2(S0, Si, t, r, a1, b, rho, m, sigma, lam)
            #vol2 = 0.4**2  # testing
            alpha[i] = 0.5 * (vol2 * Si**2 / dS**2 - r * Si / dS)
            beta[i] = -(vol2 * Si**2 / dS**2 + r)
            gamma[i] = 0.5 * (vol2 * Si**2 / dS**2 + r * Si / dS)

        # Dirichlet boundaries
        if option_type == "put":
            V[0], V[M] = K, 0.0
        else:
            V[0], V[M] = 0.0, S_max - K

        # build RHS: b = (I + (1-θ)dt_n A) V^n + boundary injections
        b_vec = V.copy()
        for i in range(1, M):
            b_vec[i] += (1 - theta) * dt_n * (alpha[i] * V[i - 1] + beta[i] * V[i] + gamma[i] * V[i + 1])
        # inject Dirichlet into neighbors
        b_vec[1] += theta * dt_n * alpha[1] * V[0]
        b_vec[M - 1] += theta * dt_n * gamma[M - 1] * V[M]

        # Now M = I - θ dt_n A has
        #   M[i,i-1] = -theta dt_n * alpha[i]
        #   M[i,i]   =  1    - theta dt_n * beta[i]
        #   M[i,i+1] = -theta dt_n * gamma[i]
        # but we absorb dt_n and θ into alpha/beta/gamma temporarily:
        alpha_psor = theta * dt_n * alpha
        beta_psor = 1.0 - theta * dt_n * beta
        gamma_psor = theta * dt_n * gamma

        # solve LCP by PSOR on interior
        V = _psor_step(alpha_psor, beta_psor, gamma_psor, b_vec, payoff, option_type="put", omega=omega, tol=tol)

        # store one-step-forward solution for theta
        if n == 1:
            V_one_step = V.copy()

    # 4. Price at S0
    price = float(V[i_S0])

    # 5. Greeks
    delta = (V[i_S0 + 1] - V[i_S0 - 1]) / (2 * dS)
    gamma_val = (V[i_S0 + 1] - 2 * V[i_S0] + V[i_S0 - 1]) / dS**2
    theta = ((V_one_step[i_S0] - V[i_S0]) / dt) if V_one_step is not None else np.nan

    return price, delta, gamma_val, theta, V, S


def generate_american_put_data_set(folder, label, N_data):
    # Parameters
    S0 = 1

    T = 1.0
    K_vals = np.random.uniform(0.5, 1.5, N_data)  # strike prices
    r_vals = np.random.uniform(0.001, 0.06, N_data)  # risk-free rate
    # SVI parameters
    a1_vals = np.random.uniform(0.001, 0.01, N_data)  # a1>0
    b_vals = np.random.uniform(0.05, 0.4, N_data)  # b>0
    rho_vals = np.random.uniform(-0.8, 0.8, N_data)  # |rho|<1
    m_vals = np.random.uniform(-0.5, 1.0, N_data)  # m unlimited
    sigma_vals = np.random.uniform(0.05, 1.0, N_data)  # sigma>0
    lam_vals = np.random.uniform(0.00, 1.0, N_data)  # lam>0

    price_vals = np.zeros(N_data)
    delta_vals = np.zeros(N_data)
    gamma_vals = np.zeros(N_data)
    theta_vals = np.zeros(N_data)
    time_vals = np.zeros(N_data)

    for i in range(N_data):
        start_time = time.time()
        price, delta, gamma, theta, V, S = price_american_option_PSOR(S0, K_vals[i], r_vals[i], T, a1_vals[i], b_vals[i], rho_vals[i], m_vals[i], sigma_vals[i], lam_vals[i], option_type="put")
        price_vals[i] = price
        delta_vals[i] = delta
        gamma_vals[i] = gamma
        theta_vals[i] = theta
        #if i % 1 == 0:
        print(f"Progress: {i+1}/{N_data}")
        #print(price_vals)
        elapsed_time = time.time() - start_time
        time_vals[i] = elapsed_time

    # save data to a CSV file
    data = np.column_stack((K_vals, r_vals, a1_vals, b_vals, rho_vals, m_vals, sigma_vals, lam_vals, price_vals, delta_vals, gamma_vals, theta_vals, time_vals))
    file_path = f"{folder}/american_put_data_{label}.csv"
    header = "K,r,a1,b,rho,m,sigma,lam,price,delta,gamma,theta,run_time"
    np.savetxt(file_path, data, delimiter=",", header=header, comments="")
    print(f"Data saved to: {file_path}")


