import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from SVI_vol import *
from scipy.linalg import solve_banded
import time
from numdifftools.multicomplex import Bicomplex
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def calc_greek_by_bump(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type, S_max_mult=5.0, M=500, N=500, method="CN"):

    V_mid, V_down, V_up = None, None, None
    if method == "CN":
        price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V = price_american_option_CN(
            S0, K, r, T, a1, b, rho, m, sigma, lam, option_type, S_max_mult=S_max_mult, M=M, N=N
        )
        V_mid = all_V[0, :]
        bump = 1e-5
        price_up, delta_up, gamma_up, theta_up, price_grid_up, delta_grid_up, gamma_grid_up, theta_grid_up, S_up, all_V = price_american_option_CN(
            S0, K, r, T, a1, b, rho, m, sigma, lam, option_type, S_max_mult=S_max_mult, M=M, N=N, bump=bump
        )
        V_up = all_V[0, :]
        price_down, delta_down, gamma_down, theta_down, price_grid_down, delta_grid_down, gamma_grid_down, theta_grid_down, S_down, all_V = price_american_option_CN(
            S0, K, r, T, a1, b, rho, m, sigma, lam, option_type, S_max_mult=S_max_mult, M=M, N=N, bump=-bump
        )
        V_down = all_V[0, :]
    delta_grid = (V_up - V_down) / (2 * bump)
    gamma_grid = (V_up - 2 * V_mid + V_down) / (bump**2)

    return delta_grid, gamma_grid, S


def price_american_option_CN(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type, S_max_mult=5.0, M=500, N=500, bump=0):
    # 1) Build grid in S and time
    S_max = S_max_mult * K
    S_grid, dS, i_S0 = make_grid_with_S0_on_node(S0, S_max, M)
    S_grid = S_grid + bump
    dt = T / N

    # 2) Terminal payoff at t=T
    if option_type.lower() == "put":
        payoff_full = np.maximum(K - S_grid, 0.0)
    else:
        payoff_full = np.maximum(S_grid - K, 0.0)
    V_full = payoff_full.copy()

    # 3) Storage: rows=time, cols=space
    V_all = np.zeros((N + 1, M + 1))
    V_all[N, :] = V_full

    # 4) Interior spatial indices and identity
    interior = np.arange(2, M - 1)
    n_int = interior.size
    In = sp.eye(n_int)

    # 5) Backward Euler loop (projected CN)
    for n in range(N - 1, -1, -1):
        t = n * dt  # current time

        # 5a) Dirichlet boundaries
        if option_type.lower() == "put":
            V_full[0], V_full[-1] = K, 0.0
        elif option_type.lower() == "call":
            V_full[0] = 0.0
            V_full[-1] = S_grid[-1] - K

        # 5b) Assemble local-vol operator A(t)
        A = sp.lil_matrix((n_int, n_int))
        # we'll need coefficients at the two boundary rows for injection
        # left-boundary i=2 -> row 0
        # right-boundary i=M-2 -> row n_int-1
        for row, i in enumerate(interior):
            S_i = S_grid[i]
            vol2 = local_vol2(S0, S_i, t, r, a1, b, rho, m, sigma, lam)
            # vol2 = 0.03  # testing
            c2 = 0.5 * vol2 * S_i**2
            c1 = r * S_i
            # second-derivative five-point
            if row - 2 >= 0:
                A[row, row - 2] = -c2 / (12 * dS**2)
            if row - 1 >= 0:
                A[row, row - 1] = +16 * c2 / (12 * dS**2)
            A[row, row] = -30 * c2 / (12 * dS**2)
            if row + 1 < n_int:
                A[row, row + 1] = +16 * c2 / (12 * dS**2)
            if row + 2 < n_int:
                A[row, row + 2] = -c2 / (12 * dS**2)
            # first-derivative five-point
            if row - 2 >= 0:
                A[row, row - 2] += +c1 / (12 * dS)
            if row - 1 >= 0:
                A[row, row - 1] += -8 * c1 / (12 * dS)
            if row + 1 < n_int:
                A[row, row + 1] += +8 * c1 / (12 * dS)
            if row + 2 < n_int:
                A[row, row + 2] += -c1 / (12 * dS)
            # center term
            A[row, row] += -r
        A = A.tocsr()

        # 5c) Crank-Nicolson matrices
        M1 = In - 0.5 * dt * A
        M2 = In + 0.5 * dt * A

        # 5d) Form RHS and inject boundaries
        V_int = V_full[interior]
        B = M2.dot(V_int)
        # left injection (row=0 corresponds to i=2)
        # coefficients at i=2
        S2 = S_grid[interior[0]]
        vol2 = local_vol2(S0, S2, t, r, a1, b, rho, m, sigma, lam)
        # vol2 = 0.03  # testing
        c2 = 0.5 * vol2 * S2**2
        c1 = r * S2
        # second-derivative coeffs
        coef_im2 = -c2 / (12 * dS**2)
        coef_im1 = +16 * c2 / (12 * dS**2)
        # first-derivative coeffs
        coef_f_im2 = +c1 / (12 * dS)
        coef_f_im1 = -8 * c1 / (12 * dS)
        B[0] += -(coef_im2 + coef_f_im2) * V_full[0]
        B[0] += -(coef_im1 + coef_f_im1) * V_full[1]
        # right injection (row = n_int-1 => i = M-2)
        S_im2 = S_grid[interior[-1]]
        vol2 = local_vol2(S0, S_im2, t, r, a1, b, rho, m, sigma, lam)
        # vol2 = 0.03  # testing
        c2 = 0.5 * vol2 * S_im2**2
        c1 = r * S_im2
        coef_ip2 = -c2 / (12 * dS**2)
        coef_ip1 = +16 * c2 / (12 * dS**2)
        coef_f_ip2 = -c1 / (12 * dS)
        coef_f_ip1 = +8 * c1 / (12 * dS)
        B[-1] += -(coef_ip1 + coef_f_ip1) * V_full[-2]
        B[-1] += -(coef_ip2 + coef_f_ip2) * V_full[-1]

        # 5e) Solve and project
        V_new = spla.spsolve(M1, B)
        V_full[interior] = V_new
        V_full = np.maximum(V_full, payoff_full)

        # 5f) Store
        V_all[n, :] = V_full

    # Greeks at t=0
    V_t0 = V_all[0]
    delta_t0 = np.zeros_like(V_t0)
    gamma_t0 = np.zeros_like(V_t0)
    theta_t0 = np.zeros_like(V_t0)
    # central differences in S
    delta_t0[1:-1] = (V_t0[2:] - V_t0[:-2]) / (2 * dS)
    gamma_t0[1:-1] = (V_t0[2:] - 2 * V_t0[1:-1] + V_t0[:-2]) / (dS**2)
    # time difference for theta
    theta_t0[:] = (V_all[1] - V_all[0]) / dt
    price_t0 = V_t0

    # spot value
    price = float(V_t0[i_S0])
    delta = float(delta_t0[i_S0])
    gamma = float(gamma_t0[i_S0])
    theta = float(theta_t0[i_S0])

    return price, delta, gamma, theta, price_t0, delta_t0, gamma_t0, theta_t0, S_grid, V_all


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
                # V_new = update
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


def price_american_option_PSOR(S0, K, r, T, a1, b, rho, m, sigma, lam, option_type, S_max_mult=5.0, M=500, N=500, omega=1.2, tol=1e-8, refine_theta=False, dt_theta=1e-5):  # SVI ignored here
    # 1. Spatial grid in S
    # usage inside your solver

    S, dS, i_S0 = make_grid_with_S0_on_node(S0, S_max_mult * K, M)

    '''
    print("np.max(S)", np.max(S))
    print("np.min(S)", np.min(S))
    print("len(S)", len(S))
    print("S0", S0)
    print("i_S0", i_S0)
    '''

    # Initialize arrays to record the full grid of S and V (option values at all time steps)
    # all_S contains the spatial grid S repeated across all time layers
    all_V = np.empty((M + 1, N + 1))

    # 2. Payoff and initial condition
    if option_type == "put":
        payoff = np.maximum(K - S, 0.0)
    elif option_type == "call":
        payoff = np.maximum(S - K, 0.0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")
    V = payoff.copy()
    all_V[:, -1] = V
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
            vol2 = local_vol2(S0, Si, t, r, a1, b, rho, m, sigma, lam)
            # vol2 = 0.03  # testing
            alpha[i] = 0.5 * (vol2 * Si**2 / dS**2 - r * Si / dS)
            beta[i] = -(vol2 * Si**2 / dS**2 + r)
            gamma[i] = 0.5 * (vol2 * Si**2 / dS**2 + r * Si / dS)

        # Dirichlet boundaries
        if option_type == "put":
            V[0], V[M] = K - S[0], 0.0
        elif option_type == "call":
            V[0], V[-10 : M + 1] = 0.0, S[-10 : M + 1] - K
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
        all_V[:, n] = V  # store the option values at this time step
        # all_S[:, n] = S

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
    # delta_grid[2:-2] = (-V[4:] + 8*V[3:-1] - 8*V[1:-3] + V[0:-4]) / (12*dS)
    # gamma_grid[2:-2] = (-V[4:] + 16*V[3:-1] - 30*V[2:-2] + 16*V[1:-3] - V[0:-4]) / (12*dS**2)

    # 5b. Theta: either the old crude or the refined small-dt step
    if refine_theta and V_one_step is not None:
        # if you have a helper that returns a full theta‐grid:
        # V, S, i_S0, S0, K, r, a1, b, rho, m, sigma, lam, payoff, option_type, dS, omega, tol, dt_theta
        Theta, theta_grid = _refine_theta_PSOR(V, S, i_S0, S0, K, r, a1, b, rho, m, sigma, lam, payoff, option_type, dS, omega, tol, dt_theta)
    else:
        if V_one_step is not None:
            theta_grid = (V_one_step - V) / dt
        else:
            theta_grid = np.full_like(V.real, np.nan)
        Theta = theta_grid[i_S0]

    return price, delta, gamma, Theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V


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


def generate_american_put_data_set(folder, label, N_data, data_type="train"):
    # Parameters
    S0 = 1

    T = 1.0
    if type == "train":
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
        lam_vals = np.random.uniform(0.00, 1.0, N_data)  # lam>0
    elif type == "test":
        K_vals = np.random.uniform(0.9, 1.1, N_data)
        r_vals = np.random.uniform(0.01, 0.05, N_data)
        # SVI parameters
        a1_vals = np.random.uniform(0.005, 0.015, N_data)
        b_vals = np.random.uniform(0.05, 0.25, N_data)
        rho_vals = np.random.uniform(-0.3, 0.7, N_data)
        m_vals = np.random.uniform(-0.1, 0.5, N_data)
        sigma_vals = np.random.uniform(0.1, 0.9, N_data)
        lam_vals = np.random.uniform(0.1, 0.9, N_data)

    price_vals = np.zeros(N_data)
    delta_vals = np.zeros(N_data)
    gamma_vals = np.zeros(N_data)
    theta_vals = np.zeros(N_data)
    time_vals = np.zeros(N_data)

    for i in range(N_data):
        start_time = time.time()
        price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V = price_american_option_PSOR(
            S0, K_vals[i], r_vals[i], T, a1_vals[i], b_vals[i], rho_vals[i], m_vals[i], sigma_vals[i], lam_vals[i], option_type="put", M=1000, N=1000
        )
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
    file_path = f"{folder}/american_put_{data_type}_data_{label}.csv"

    header = "K,r,a1,b,rho,m,sigma,lam,price,delta,gamma,theta,run_time"
    np.savetxt(file_path, data, delimiter=",", header=header, comments="")
    print(f"Data saved to: {file_path}")


def generate_american_put_precision_data(folder):
    a1, b, rho, m, sigma = 0.01, 0.15, 0.2, 0.2, 0.5
    r = 0.03
    lam = 0.5
    K = 1.0
    base_params = dict(a1=a1, b=b, rho=rho, m=m, sigma=sigma, r=r, lam=lam, K=K)
    n = 20
    param_range = {
        "a1": np.linspace(0.00, 0.02, n),
        "b": np.linspace(0.0, 0.3, n),
        "rho": np.linspace(-0.4, 0.8, n),
        "m": np.linspace(-0.2, 0.6, n),
        "sigma": np.linspace(0.00, 1.0, n),
        "lam": np.linspace(0.00, 1.0, n),
        "r": np.linspace(0.00, 0.06, n),
        "K": np.linspace(0.85, 1.15, n),
    }

    # for each param, generate the values of price and greeks
    for param_name, param_vals in param_range.items():
        print(f"Processing parameter: {param_name}")
        price_vals = np.zeros(n)
        delta_vals = np.zeros(n)
        gamma_vals = np.zeros(n)
        theta_vals = np.zeros(n)
        for i, param_val in enumerate(param_vals):
            print(f"Processing {param_name}={param_val} ({i+1}/{n})")
            p = base_params.copy()
            p[param_name] = param_val
            # S0, K, r, T, a1, b, rho, m, sigma, lam, option_type, S_max_mult=5.0, M=500, N=500, omega=1.2, tol=1e-8, refine_theta=Fa
            price, delta, gamma, theta, price_grid, delta_grid, gamma_grid, theta_grid, S, all_V = price_american_option_PSOR(
                S0=1.0, K=p["K"], T=1.0, r=p["r"], a1=p["a1"], b=p["b"], rho=p["rho"], m=p["m"], sigma=p["sigma"], lam=p["lam"], option_type="put", M=1000, N=1000
            )
            price_vals[i] = price
            delta_vals[i] = delta
            gamma_vals[i] = gamma
            theta_vals[i] = theta
        # save the results to a CSV file
        data = np.column_stack((param_vals, price_vals, delta_vals, gamma_vals, theta_vals))
        file_path = f"{folder}/american_put_{param_name}_sensitivity.csv"
        header = f"{param_name},price,delta,gamma,theta"
        np.savetxt(file_path, data, delimiter=",", header=header, comments="")
        print(f"Data saved to: {file_path}")
