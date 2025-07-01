from vol_surface_plot import *
from variance_swap_plot import *
from american_put_plot import *

def main():
    print("plotting figures~")

    # SVI surface and local vol
    #plot_illustrate_svi_curve()

    #plot_illustrative_vol_surface_local_vol()

    try_to_fit_SVI_surface()

    # variance swap

    #plot_Kvar_versus_SVI_params()

    # LML + GPR
    #plot_variance_swap_GPR_fitting()


    # american put

    # sample solution of V(S), compare with BS
    #plot_american_price_solution()

    # heat map of V(S,t) and Gamma(S, t)
    # then V(S) and Gamma (S) for different r, to show discontinuity is from early exercise driven by interest rate
    #plot_american_price_solution_per_r()

    # V, delta, gamma, theta versus 6 SVI params + K and r ( need precision run)
    #plot_american_params_sensitivity()
    #plot_american_params_sensitivity_K()

    # LML
    #plot_american_put_LML()

    # GPR
    #plot_american_put_GPR_fitting()







if __name__ == "__main__":
    main()