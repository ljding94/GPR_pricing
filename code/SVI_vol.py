import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def calc_raw_SVI_skew_T1(k, a1, b, rho, m, sigma):
    """
    Calculate the SVI total implied variance for log-moneyness k.
    Assumes T = 1 so that total variance = implied volatility^2.
    """
    a = a1 - b * sigma * np.sqrt(1 - rho * rho)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def calc_raw_SVI_surface(k, T, a1, b, rho, m, sigma, lam):
    """
    Calculate the SVI total implied variance for log-moneyness k.
    Assumes T = 1 so that total variance = implied volatility^2.
    """
    a = a1 - b * sigma * np.sqrt(1 - rho * rho)
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
    # w = w * (1 - np.exp(-lam * T)) / (1 - np.exp(-lam))
    w = w * T * np.exp(lam * (1 - T))
    return w


def calc_raw_SVI_surface_derivative(k, T, a1, b, rho, m, sigma, lam):
    w_T = np.exp(lam * (1 - T)) * (1 - T * lam) * calc_raw_SVI_skew_T1(k, a1, b, rho, m, sigma)

    fT = T * np.exp(lam * (1 - T))
    w_k = b * (rho + (k - m) / np.sqrt((k - m) ** 2 + sigma**2)) * fT

    w_kk = b * sigma**2 / np.power((k - m) ** 2 + sigma**2, 1.5) * fT
    return w_T, w_k, w_kk


def local_var(k, T, a1, b, rho, m, sigma, lam):
    """
    Calculate the local variance using the SVI parameters.
    """
    w_T, w_k, w_kk = calc_raw_SVI_surface_derivative(k, T, a1, b, rho, m, sigma, lam)
    w = calc_raw_SVI_surface(k, T, a1, b, rho, m, sigma, lam)
    w = max(w, 1e-10)
    wk1_term = -k / w * w_k
    wk2_term = 0.25 * (-0.25 - 1 / w + k**2 / w**2) * w_k**2
    wkk_term = 0.5 * w_kk
    vL = w_T / (1 + wk1_term + wk2_term + wkk_term)

    # sigma_loc = np.sqrt(vL)
    return max(vL, 1e-10)


def local_vol2(S0, S, t, r, a1, b, rho, m, sigma, lam):
    """
    Calculate the local volatility using the SVI parameters.
    """
    # Calculate the forward price
    K = S
    T = t
    F = S0 * np.exp(r * T)
    k = np.log(K / F)
    vL = local_var(k, T, a1, b, rho, m, sigma, lam)
    return vL




def vol_surface(args, type="SVI"):
    """
    Returns a function that calculates volatility surface given strike K and time T.

    Parameters:
    args: tuple - SVI parameters (a1, b, rho, m, sigma, lam)
    type: str - Type of volatility model, currently only "SVI" is implemented

    Returns:
    function - A function that takes K, T as inputs and returns volatility
    """
    if type == "SVI":

        def svi_vol(K, T):
            S0, r, a1, b, rho, m, sigma, lam = args
            # Convert to log-moneyness
            F = S0 * np.exp(r * T)
            k = np.log(K / F)
            w = calc_raw_SVI_surface(k, T, a1, b, rho, m, sigma, lam)
            return np.sqrt(w / T)

        return svi_vol
    elif type == "flat":

        def flat_vol(K, T):
            vol = args[0]
            return vol

        return flat_vol
    elif type == "Derman":

        def derman_vol(K, T):
            S0, vol_atm, slope = args  # Derman's example, S0=100, vol_atm=0.1, slope = -0.05
            vol = vol_atm + slope * (K / S0 - 1)
            return vol

        return derman_vol

    else:
        raise ValueError(f"Volatility model type '{type}' not implemented")


def bs_price(S, K, T, r, vol, type):
    if T <= 0:
        # At expiry, price equals the intrinsic value
        if type.lower() == "call":
            return max(S - K, 0)
        elif type.lower() == "put":
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    if type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Calculate option Greeks using Black-Scholes formulas
    pdf_d1 = norm.pdf(d1)
    if type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * vol / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    elif type.lower() == "put":
        delta = norm.cdf(d1) - 1
        theta = (-S * pdf_d1 * vol / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    gamma = pdf_d1 / (S * vol * np.sqrt(T))

    # Optionally, print or log the Greeks for debugging
    #print(f"Delta: {delta}, Gamma: {gamma}, Theta: {theta}")
    return price, delta, gamma, theta


def build_binomial_tree(S0, T, r, N, vol_arg, vol_type):  # a1, b, rho, m, sigma, lam):
    dt = T / N
    # svi_params = (a1, b, rho, m, sigma, lam)
    vol_surface_func = vol_surface(vol_arg, type=vol_type)
    # tree initialization
    # stock price tree
    S = [np.zeros(n + 1) for n in range(N + 1)]
    # up movement probability
    P = [np.zeros(n + 1) for n in range(N + 1)]
    # Arrow-Debreu prices
    lbd = [np.zeros(n + 1) for n in range(N + 1)]

    S[0][0] = S0
    lbd[0][0] = 1.0

    erdt = np.exp(r * dt)
    per_erdt = np.exp(-r * dt)

    # build the tree
    # ref: Derman, Kani, "Riding on a smile"
    # known S(n-1) , lbd(n-1) , -> S(n), P(n-1)  -> lambda(n)
    for n in range(1, N + 1):
        # some notation to map with the paper

        # si = S[n-1][i]
        # Si = S[n][i]
        # Fi  = si*e^{r dt} = S[n-1][i]*np.exp(r*dt) = P[n-1][i]*S[n][i+1] + (1-P[n-1][i])*S[n][i]
        # lbdi = lbd[n-1][i]
        # pi = P[n-1][i]

        # 1. find the stock price at the middle of the tree
        # there are n+1 nodes at time n
        # if n + 1 is odd, n is even

        if n % 2 == 0:
            i = int(n / 2)  # Center node
            S[n][i] = S0

        else:
            # n is odd, n + 1 is even
            i = int((n - 1) / 2)
            Sigma = np.sum([lbd[n - 1][j] * (erdt**n * S[n - 1][j] - S[n - 1][i]) for j in range(i + 1, n)])
            vol = vol_surface_func(S0, n * dt)  # strike S0, maturity n*dt
            # bs_call = np.exp(-r*n*dt)*bs_price(S0*np.exp(r*n*dt), S0, dt, r, vol, "call")
            bs_call = bs_price(S0, S0, n * dt, r, vol, "call")
            print(f"bs_call: {bs_call}, spot: {S0}, strike: {S0}, maturity: {n*dt}, vol: {vol}, r: {r}")
            S[n][i + 1] = S0 * (erdt * bs_call + lbd[n - 1][i] * S0 - Sigma)
            S[n][i + 1] /= lbd[n - 1][i] * S[n - 1][i] * erdt**n - erdt * bs_call + Sigma

            S[n][i] = S0 * S0 / S[n][i + 1]

        # now we have the stock price at the center of the tree
        print("done filling the middle of the tree")
        # 2. find the stock in the uppder half of the tree at n and find P[n-1]
        # calc price of S[n][i+1]
        i_up_start = int(n / 2) if n % 2 == 0 else int((n - 1) / 2 + 1)
        # i_up_start is know, will calculate i_up_start +1 first
        print("i_up_start", i_up_start)
        for i in range(i_up_start, n):
            # find the stock price at the upper half of the tree
            Sigma = np.sum([lbd[n - 1][j] * (erdt**n * S[n - 1][j] - S[n - 1][i]) for j in range(i + 1, n)])  # may need to check if sum to n is correct
            vol = vol_surface_func(S[n - 1][i], n * dt)  # strike S[n-1][i], maturity n*dt
            # bs_call = np.exp(-r*n*dt)*bs_price(S0*np.exp(r*n*dt), S[n - 1][i], dt, r, vol, "call")
            bs_call = bs_price(S0, S[n - 1][i], n * dt, r, vol, "call")
            print(f"bs_call: {bs_call}, strike: {S[n - 1][i]}, maturity: {n*dt}, vol: {vol}")

            C_term = erdt * bs_call - Sigma
            lbd_term = lbd[n - 1][i] * (erdt * S[n - 1][i] - S[n][i])
            S[n][i + 1] = (S[n][i] * C_term - lbd_term * S[n - 1][i]) / (C_term - lbd_term)

            # find the transition probability
            print("uppder half")
            print("S", S)
            print(f"C: n: {n}, i: {i}, S[n][i]: {S[n][i]} S[n][i+1]: {S[n][i+1]}, C_term: {C_term}, lbd_term: {lbd_term}")
            # P[n-1][i] = (S[n-1][i]*erdt - S[n][i])/(S[n][i+1] - S[n][i])

        # 3. find the stock in the lower half of the tree at n and find P[n-1]
        i_down_start = int(n / 2) if n % 2 == 0 else int((n - 1) / 2)
        # i_down_start is know, will calculate i_down_start -1 first
        print("i_down_start", i_down_start)
        for i in range(i_down_start - 1, -1, -1):
            # find the stock price at the lower half of the tree
            Sigma = np.sum([lbd[n - 1][j] * (S[n - 1][i] - erdt**n * S[n - 1][j]) for j in range(0, i)])
            vol = vol_surface_func(S[n - 1][i], n * dt)  # strike S[n-1][i], maturity n*dt
            bs_put = np.exp(-r * n * dt) * bs_price(S0 * np.exp(r * n * dt), S[n - 1][i], dt, r, vol, "put")
            print(f"bs_put: {bs_put}, strike: {S[n - 1][i]}, maturity: {n*dt}, vol: {vol}")
            P_term = erdt * bs_put - Sigma
            lbd_term = lbd[n - 1][i] * (erdt * S[n - 1][i] - S[n][i + 1])
            S[n][i] = (S[n][i + 1] * P_term + lbd_term * S[n - 1][i]) / (P_term + lbd_term)
            # find the transition probability
            print("lower half")
            print("S", S)
            print(f"P: n: {n},  i: {i}, S[n][i]: {S[n][i]}, S[n][i+1]: {S[n][i+1]}, P_term: {P_term}, lbd_term: {lbd_term}")
            # P[n-1][i] = (S[n-1][i]*erdt - S[n][i])/(S[n][i+1] - S[n][i])

        # update  P
        for i in range(n):
            P[n - 1][i] = (S[n - 1][i] * erdt - S[n][i]) / (S[n][i + 1] - S[n][i])
            print(f"P[{n-1}][{i}]", P[n - 1][i])

        # 4. update the Arrow-Debreu price on the tree at n using P[n-1] and lbd[n-1]
        lbd[n][0] = lbd[n - 1][0] * (1 - P[n - 1][0]) * per_erdt
        lbd[n][n] = lbd[n - 1][n - 1] * P[n - 1][n - 1] * per_erdt
        # everything else in the middle has two part
        for i in range(1, n):
            lbd[n][i] = (lbd[n - 1][i - 1] * P[n - 1][i - 1] + lbd[n - 1][i] * (1 - P[n - 1][i])) * per_erdt

        print("after the layer's update")
        print("n", n)
        print("S", S)
        print("P", P)
        print("lbd", lbd)
        print("==========================")

    return S, P, lbd
