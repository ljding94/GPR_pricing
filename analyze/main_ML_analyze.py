#!/opt/homebrew/bin/python3
from ML_analyze import *
import time
import random


def main():

    print("analyzing data using ML model")
    #folder = "../data/data_pool"
    #folder = "../data/20250430"
    folder = "../data/20250505_vs"
    folder = "../data/20250505"
    # rand_max = 1000
    # filter by inK

    #product = "variance_swap"
    #data_train = read_variance_swap_data(folder, "train")
    #data_test = read_variance_swap_data(folder, "test")

    product = "american_put"
    data_train = read_american_put_data(folder, "train", 99)
    data_test =  read_american_put_data(folder, "test", 18)
    print("np.shape(data_train)", len(data_train[0]))
    print("np.shape(data_test)", len(data_test[0]))
    #print("data", data[:10])
    params, params_tex, params_name, targets, targets_tex, targets_name = data_train
    #targe_distribution(folder, data) # plot distrubution of target
    print(params[:5])
    print(targets[:5])

    # Generate a random permutation of the indices
    # Set a seed for reproducibility
    #np.random.seed(42)
    #perm = np.random.permutation(params.shape[0])

    # Shuffle params and target in the same order
    #params_shuffled = params[perm]
    #target_shuffled = target[perm]

    # If you want to update your data tuple:
    #data_shuffled = [params_shuffled, params_tex, params_name, target_shuffled, target_tex, target_name]


    perc_train = 1.0
    GaussianProcess_optimization(folder, data_train, perc_train, product)

    #all_feature_names, all_feature_mean, all_feature_std, all_gp_per_feature = read_gp_and_params_stats(folder, "_all")


    GaussianProcess_prediction(folder, data_test, product)


    #plot_data_distribution(data_train[3], data_test[3])


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")