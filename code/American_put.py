import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from SVI_vol import *
from scipy.linalg import solve_banded
import time
from numdifftools.multicomplex import Bicomplex


def _psor_step(alpha, beta, gamma, b, payoff, option_type, omega=1.2, tol=1e-8, maxit=2000):
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
            elif option_type == "call":
                #V_new = update
                V_new = max(payoff[i], update)
            else:
                raise ValueError("option_type must be 'put' or 'call'")
            err = max(err, abs(V_new - V[i]))
            V[i] = V_new
        if err < tol:
            break
    return V


def _refine_theta_PSOR(V, S, i_S0, S0, K, r, a1, b, rho, m, sigma, lam, payoff, option_type, dS, omega, tol, dt_theta):
    """
    Perform one extra Crank–Nicolson+PSOR step of size dt_theta at t=0
    and return (V_small[i_S0] - V[i_S0]) / dt_theta.
    """
    M = len(S) - 1

    # 1) build A-coeffs at t=0
    alpha0 = np.zeros(M + 1)
    beta0 = np.zeros(M + 1)
    gamma0 = np.zeros(M + 1)
    for i in range(1, M):
        vol2 = local_vol2(S0, S[i], 0.0, r, a1, b, rho, m, sigma, lam)
        alpha0[i] = 0.5 * (vol2 * S[i] ** 2 / dS**2 - r * S[i] / dS)
        beta0[i] = -(vol2 * S[i] ** 2 / dS**2 + r)
        gamma0[i] = 0.5 * (vol2 * S[i] ** 2 / dS**2 + r * S[i] / dS)

    # 2) Dirichlet BCs at t=0
    if option_type == "put":
        V[0], V[M] = K, 0.0
    else:
        V[0], V[M] = 0.0, S[-1] - K

    # 3) CN parameters
    theta_cn = 0.5
    a_ps = theta_cn * dt_theta * alpha0
    b_ps = 1.0 - theta_cn * dt_theta * beta0
    g_ps = theta_cn * dt_theta * gamma0

    # 4) build RHS for the small step
    b_small = V.copy()
    for i in range(1, M):
        b_small[i] += (1 - theta_cn) * dt_theta * (alpha0[i] * V[i - 1] + beta0[i] * V[i] + gamma0[i] * V[i + 1])
    b_small[1] += theta_cn * dt_theta * alpha0[1] * V[0]
    b_small[M - 1] += theta_cn * dt_theta * gamma0[M - 1] * V[M]

    # 5) one PSOR solve
    V_small = _psor_step(a_ps, b_ps, g_ps, b_small, payoff, option_type=option_type, omega=omega, tol=tol)

    # 6) return refined theta
    theta = (V_small[i_S0] - V[i_S0]) / dt_theta
    theta_grid = (V_small - V) / dt_theta
    return theta, theta_grid


def make_grid_with_S0_on_node(S0, S_max, M):
    # 1) raw step
    dS = S_max / M

    # 2) index nearest to S0
    i0 = int(round(S0 / dS))

    # 3) lower bound so that S[i0] == S0, ensure non-negative grid start
    S_min = max(S0 - i0 * dS, 0)

    # 4) build grid
    S = S_min + np.arange(M + 1) * dS
    return S, dS, i0


def price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type, S_max_mult=2.0, M=400, N=400, omega=1.2, tol=1e-8, refine_theta=False, dt_theta=1e-5):  # SVI ignored here
    # 1. Spatial grid in S
    # usage inside your solver
    S, dS, i_S0 = make_grid_with_S0_on_node(S0, S_max_mult * K, M)

    print("np.max(S)", np.max(S))
    print("np.min(S)", np.min(S))
    print("len(S)", len(S))
    print("S0", S0)
    print("i_S0", i_S0)

    # 2. Payoff and initial condition
    if option_type == "put":
        payoff = np.maximum(K - S, 0.0)
    elif option_type == "call":
        payoff = np.maximum(S - K, 0.0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")
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
        # dt_n, theta = dt, 0.5

        # build A-coeffs at this layer (flat vol test; replace with local_var if you like)
        for i in range(1, M):
            Si = S[i]
            # vol2 = local_vol2(S0, Si, t, r, a1, b, rho, m, sigma, lam)
            vol2 = 0.03  # testing
            alpha[i] = 0.5 * (vol2 * Si**2 / dS**2 - r * Si / dS)
            beta[i] = -(vol2 * Si**2 / dS**2 + r)
            gamma[i] = 0.5 * (vol2 * Si**2 / dS**2 + r * Si / dS)

        # Dirichlet boundaries
        if option_type == "put":
            V[0], V[M] = K - S[0], 0.0
        elif option_type == "call":
            V[0], V[M] = 0.0, S[M] - K
        else:
            raise ValueError("option_type must be 'put' or 'call'")

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
        V = _psor_step(alpha_psor, beta_psor, gamma_psor, b_vec, payoff, option_type, omega=omega, tol=tol)

        # store one-step-forward solution for theta
        if n == 1:
            V_one_step = V.copy()

    # 4. Price at S0
    price = float(V[i_S0])

    # 5. Greeks
    delta = (V[i_S0 + 1] - V[i_S0 - 1]) / (2 * dS)
    gamma = (V[i_S0 + 1] - 2 * V[i_S0] + V[i_S0 - 1]) / dS**2

    # full grid:
    price_grid = V.copy()
    delta_grid = np.full(V.shape, np.nan)
    gamma_grid = np.full(V.shape, np.nan)
    delta_grid[1:M] = (V[2:] - V[:-2]) / (2 * dS)
    gamma_grid[1:M] = (V[2:] - 2 * V[1:M] + V[:-2]) / dS**2

    # 4th-order stencil
    # make sure i >= 2 and i <= M-2
    # delta = (-V[i+2] + 8*V[i+1] - 8*V[i-1] + V[i-2]) / (12*dS)
    # gamma = (-V[i+2] + 16*V[i+1] - 30*V[i] + 16*V[i-1]- V[i-2]) / (12*dS**2)

    # 5b. Theta: either the old crude or the refined small-dt step
    if refine_theta and V_one_step is not None:
        # if you have a helper that returns a full theta‐grid:
        # V, S, i_S0, S0, K, r, a1, b, rho, m, sigma, lam, payoff, option_type, dS, omega, tol, dt_theta
        theta, theta_grid = _refine_theta_PSOR(V, S, i_S0, S0, K, r, a1, b, rho, m, sigma, lam, payoff, option_type, dS, omega, tol, dt_theta)
    else:
        if V_one_step is not None:
            theta_grid = (V_one_step - V) / dt
        else:
            theta_grid = np.full_like(V.real, np.nan)
        theta = theta_grid[i_S0]

    return price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S


def _psor_step_mc(alpha, beta, gamma, b, payoff, option_type="put", omega=1.2, tol=1e-8, maxit=5000):
    M_size = len(beta)
    V = payoff.copy()  # now payoff is Bicomplex array
    for it in range(maxit):
        err = 0.0
        for i in range(1, M_size - 1):
            r_i = b[i] - (-alpha[i] * V[i - 1] + beta[i] * V[i] - gamma[i] * V[i + 1])
            update = V[i] + omega * (r_i / beta[i])
            if option_type == "put":
                V_new = Bicomplex.__max__(payoff[i], update)  # use Bicomplex max
            else:
                V_new = update
            err = max(err, abs(V_new - V[i]))
            V[i] = V_new
        if err < tol:
            break
    return V


# --- Helper: refined theta on bicomplex grid ---
def _refine_theta_PSOR_mc(V, S, i_S0, S0_bc, K, r, a1, b, rho, m, sigma, lam, payoff_bc, option_type, dS_bc, omega, tol, dt_theta):
    M = len(S) - 1
    # Build t=0 operators
    alpha0 = np.zeros(M + 1, dtype=Bicomplex)
    beta0 = np.zeros(M + 1, dtype=Bicomplex)
    gamma0 = np.zeros(M + 1, dtype=Bicomplex)
    for i in range(1, M):
        vol2 = local_vol2(S0_bc, S[i], 0.0, r, a1, b, rho, m, sigma, lam)
        alpha0[i] = Bicomplex(0, 0) + 0.5 * (vol2 * S[i] ** 2 / dS_bc**2 - r * S[i] / dS_bc)
        beta0[i] = Bicomplex(0, 0) - (vol2 * S[i] ** 2 / dS_bc**2 + r)
        gamma0[i] = Bicomplex(0, 0) + 0.5 * (vol2 * S[i] ** 2 / dS_bc**2 + r * S[i] / dS_bc)
    # BCs
    if option_type == "put":
        V[0], V[M] = Bicomplex(K, 0), Bicomplex(0, 0)
    else:
        V[0], V[M] = Bicomplex(0, 0), Bicomplex(S[-1] - K, 0)
    # tiny CN step weights
    theta_cn = 0.5
    a_ps = theta_cn * dt_theta * alpha0
    ones = np.empty_like(beta0)
    ones[:] = Bicomplex(1, 0)
    b_ps = ones - theta_cn * dt_theta * beta0
    g_ps = theta_cn * dt_theta * gamma0
    # build RHS
    b_small = V.copy()
    for i in range(1, M):
        b_small[i] += (1 - theta_cn) * dt_theta * (alpha0[i] * V[i - 1] + beta0[i] * V[i] + gamma0[i] * V[i + 1])
    b_small[1] += theta_cn * dt_theta * alpha0[1] * V[0]
    b_small[M - 1] += theta_cn * dt_theta * gamma0[M - 1] * V[M]
    # PSOR solve
    V_small = _psor_step_mc(a_ps, b_ps, g_ps, b_small, payoff_bc, option_type=option_type, omega=omega, tol=tol)
    # extract thetas
    theta_spot = (V_small[i_S0].real - V[i_S0].real) / dt_theta
    theta_grid = (V_small.real - V.real) / dt_theta
    return theta_spot, theta_grid


# --- Main pricing with bicomplex Greeks ---
def price_american_option_PSOR_mc(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type="put", S_max_mult=3.0, M=400, N=400, omega=1.2, tol=1e-8, h=1e-8, refine_theta=False, dt_theta=1e-5):
    # lift S0
    S0_bc = Bicomplex(S0 + 1j * h, h)
    # real grid
    S_max = S_max_mult * K
    S_real, dS_real, i_S0 = make_grid_with_S0_on_node(S0, S_max, M)
    # bicomplex grid
    S_bc = np.vectorize(lambda x: Bicomplex(x, 0))(S_real)
    dS_bc = Bicomplex(dS_real, 0)
    # payoff real and bicomplex
    if option_type == "put":
        payoff_real = np.maximum(K - S_real, 0.0)
    else:
        payoff_real = np.maximum(S_real - K, 0.0)
    payoff_bc = np.vectorize(lambda x: Bicomplex(x, 0))(payoff_real)
    V = payoff_bc.copy()
    # pre-alloc
    alpha = np.zeros(M + 1, dtype=Bicomplex)
    beta = np.zeros(M + 1, dtype=Bicomplex)
    gamma = np.zeros(M + 1, dtype=Bicomplex)
    dt = T / N
    V_one_step = None
    # time march
    for n in range(N - 1, -1, -1):
        t = n * dt
        dt_n, theta = (dt / 2, 1.0) if n >= N - 2 else (dt, 0.5)
        for i in range(1, M):
            vol2 = local_vol2(S0_bc, S_bc[i], t, r, a1, b, rho, m, sigma, lam)
            alpha[i] = Bicomplex(0, 0) + 0.5 * (vol2 * S_bc[i] ** 2 / dS_bc**2 - r * S_bc[i] / dS_bc)
            beta[i] = Bicomplex(0, 0) - (vol2 * S_bc[i] ** 2 / dS_bc**2 + r)
            gamma[i] = Bicomplex(0, 0) + 0.5 * (vol2 * S_bc[i] ** 2 / dS_bc**2 + r * S_bc[i] / dS_bc)
        # BCs
        if option_type == "put":
            V[0], V[M] = Bicomplex(K, 0), Bicomplex(0, 0)
        else:
            V[0], V[M] = Bicomplex(0, 0), Bicomplex(S_max - K, 0)
        # RHS build
        b_vec = V.copy()
        for i in range(1, M):
            b_vec[i] += (1 - theta) * dt_n * (alpha[i] * V[i - 1] + beta[i] * V[i] + gamma[i] * V[i + 1])
        b_vec[1] += theta * dt_n * alpha[1] * V[0]
        b_vec[M - 1] += theta * dt_n * gamma[M - 1] * V[M]
        # PSOR solve
        a_ps = theta * dt_n * alpha
        ones = np.empty_like(beta)
        ones[:] = Bicomplex(1, 0)
        b_ps = ones - theta * dt_n * beta
        g_ps = theta * dt_n * gamma
        V = _psor_step_mc(a_ps, b_ps, g_ps, b_vec, payoff_bc, option_type=option_type, omega=omega, tol=tol)
        if n == 1:
            V_one_step = V.copy()
    # extract spot & grids
    z = V[i_S0]
    price = float(z.real)
    delta = z.imag / h
    gamma = z.imag12 / (h * h)
    price_grid = V.real
    delta_grid = V.imag / h
    gamma_grid = V.imag12 / (h * h)
    # theta
    if not refine_theta or V_one_step is None:
        theta_spot = ((V_one_step[i_S0].real - z.real) / dt) if V_one_step else np.nan
        theta_grid = ((V_one_step.real - V.real) / dt) if V_one_step else np.full_like(V.real, np.nan)
    else:
        theta_spot, theta_grid = _refine_theta_PSOR_mc(V, S_bc, i_S0, S0_bc, K, r, a1, b, rho, m, sigma, lam, payoff_bc, option_type, dS_bc, omega, tol, dt_theta)
    return (price, delta, gamma, theta_spot, price_grid, delta_grid, gamma_grid, theta_grid, S_real)


def generate_precision_american_put_data_set(folder):
    pass
    # TODO: generate dataset for given SVI params, for showing


def generate_american_put_data_set(folder, label, N_data):
    # Parameters
    S0 = 1

    T = 1.0
    logK_vals = np.random.uniform(-0.1, 0.1, N_data)  # log strike prices
    # K_vals = np.random.uniform(0.5, 1.5, N_data)  # strike prices
    K_vals = S0 * np.exp(logK_vals)  # strike prices
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
        # if i % 1 == 0:
        print(f"Progress: {i+1}/{N_data}")
        # print(price_vals)
        elapsed_time = time.time() - start_time
        time_vals[i] = elapsed_time

    # save data to a CSV file
    data = np.column_stack((K_vals, r_vals, a1_vals, b_vals, rho_vals, m_vals, sigma_vals, lam_vals, price_vals, delta_vals, gamma_vals, theta_vals, time_vals))
    file_path = f"{folder}/american_put_data_{label}.csv"
    header = "K,r,a1,b,rho,m,sigma,lam,price,delta,gamma,theta,run_time"
    np.savetxt(file_path, data, delimiter=",", header=header, comments="")
    print(f"Data saved to: {file_path}")
