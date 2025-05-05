from vol_surface_plot import *
from variance_swap_plot import *
from american_put_plot import *

def main():
    print("plotting figures~")

    # SVI surface and local vol
    plot_illustrate_svi_curve()

    #plot_illustrative_vol_surface_local_vol()


    # variance swap

    #plot_Kvar_versus_SVI_params()

    # LML + GPR


    # american put

    # sample solution of V(S), compare with BS

    # V, delta, gamma, theta versus 6 SVI params + K and r ( need precision run)

    # LML + GPR







if __name__ == "__main__":
    main()