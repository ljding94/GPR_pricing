#!/opt/homebrew/bin/python3
from ML_analyze import *
import time
import random


def main():

    print("analyzing data using ML model")
    folder = "../data/data_pool"
    # rand_max = 1000
    # filter by inK

    #data_type = "american_put"
    data_type = "variance_swap"
    data = read_data(folder, data_type)
    #print("data", data[:10])
    params, params_tex, params_name, target, target_tex, target_name = data
    #targe_distribution(folder, data) # plot distrubution of target
    print(params[:5])
    print(target[:5])

    # Generate a random permutation of the indices
    # Set a seed for reproducibility
    np.random.seed(42)
    perm = np.random.permutation(params.shape[0])

    # Shuffle params and target in the same order
    params_shuffled = params[perm]
    target_shuffled = target[perm]

    # If you want to update your data tuple:
    data_shuffled = [params_shuffled, params_tex, params_name, target_shuffled, target_tex, target_name]

    perc_train = 0.4

    GaussianProcess_optimization(folder, data_shuffled, perc_train, data_type)
    #all_feature_names, all_feature_mean, all_feature_std, all_gp_per_feature = read_gp_and_feature_stats(folder, "_all")

    GaussianProcess_prediction(folder, data_shuffled, perc_train, data_type)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")