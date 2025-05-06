import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from SVI_vol import *

def calc_raw_SVI_variance(k, a, b, rho, m, sigma):
    """
    Calculate the SVI total implied variance for log-moneyness k.
    Assumes T = 1 so that total variance = implied volatility^2.
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))



def variace_swap_strike(a1, b, rho, m, sigma, S0, T, r):
    """
    Calculate the variance swap price using SVI parameters.
    """
    # Calculate the forward price
    F = S0 * np.exp(r * T)

    K_put_vals = np.linspace(0.01, 1, 50) * S0
    K_call_vals = np.linspace(1, 10, 50) * S0

    k_put_vals = np.log(K_put_vals / F)
    k_call_vals = np.log(K_call_vals / F)
    # Calculate the variance using SVI
    a = a1 - b * sigma * np.sqrt(1 - rho * rho)
    w_put_vals = calc_raw_SVI_variance(k_put_vals, a, b, rho, m, sigma)
    w_call_vals = calc_raw_SVI_variance(k_call_vals, a, b, rho, m, sigma)
    vol_call_vals = np.sqrt(w_call_vals / T)
    vol_put_vals = np.sqrt(w_put_vals / T)
    #print("vol_call_vals", vol_call_vals)

    # calculate all bs put and call prices
    bs_call_vals, delta, gamma, theta = bs_price(S0, K_call_vals, T, r, vol_call_vals, type="call")
    bs_put_vals, delta, gamma, theta = bs_price(S0, K_put_vals, T, r, vol_put_vals, type="put")

    # variance swap strike
    Kvar = 2 / T * (r * T - (1 * np.exp(r * T) - 1))
    # Calculate width for integration

    dK_put = np.diff(K_put_vals)
    # Use midpoint values or crop the longer arrays
    Kvar += np.exp(r * T) * np.sum(bs_put_vals[:-1] / K_put_vals[:-1] ** 2 * dK_put)

    # Also need to add call options part
    dK_call = np.diff(K_call_vals)
    Kvar += np.exp(r * T) * np.sum(bs_call_vals[:-1] / K_call_vals[:-1] ** 2 * dK_call)

    return Kvar


def generate_variance_swap_data(folder, N_data=1000, data_type="train"):
    """
    Generate variance swap data for a range of parameters.
    """
    # Parameters
    S0 = 1
    T = 1.0
    if data_type == "train":
        r_vals = np.random.uniform(0.00, 0.06, N_data)  # risk-free rate
        # SVI parameters
        a1_vals = np.random.uniform(0.00, 0.02, N_data)  # a1>0
        b_vals = np.random.uniform(0.00, 0.3, N_data)  # b>0
        rho_vals = np.random.uniform(-0.4, 0.8, N_data)  # |rho|<1
        m_vals = np.random.uniform(-0.2, 0.6, N_data)  # m unlimited
        sigma_vals = np.random.uniform(0.00, 1.0, N_data)  # sigma>0
    elif data_type == "test":
        r_vals = np.random.uniform(0.01, 0.05, N_data)
        # SVI parameters
        a1_vals = np.random.uniform(0.005, 0.015, N_data)
        b_vals = np.random.uniform(0.05, 0.25, N_data)
        rho_vals = np.random.uniform(-0.3, 0.7, N_data)
        m_vals = np.random.uniform(-0.1, 0.5, N_data)
        sigma_vals = np.random.uniform(0.1, 0.9, N_data)

    Kvars_vals = np.zeros(N_data)
    for i in range(N_data):
        Kvar = variace_swap_strike(a1_vals[i], b_vals[i], rho_vals[i], m_vals[i], sigma_vals[i], S0, T, r_vals[i])
        Kvars_vals[i] = Kvar
        if i % 100 == 0:
            print(f"Progress: {i}/{N_data}")
    # Save data to a CSV file
    data = np.column_stack((r_vals, a1_vals, b_vals, rho_vals, m_vals, sigma_vals, Kvars_vals, Kvars_vals))
    header = "r,a1,b,rho,m,sigma,Kvar,Kvar" # extra column as place holder
    file_path = f"{folder}/variance_swap_{data_type}_data.csv"
    np.savetxt(file_path, data, delimiter=",", header=header, comments='')
    print(f"Data saved to {file_path}")