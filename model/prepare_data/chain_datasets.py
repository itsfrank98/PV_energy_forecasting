"""Prepare files for multitarget modelling"""
import argparse
import sys
import numpy as np
sys.path.append('../..')
from utils import load_from_pickle
import os
import pandas as pd

def chain(path_to_dictionary, data_path, dst_folder, axis, cols):
    """
    Groups together the series concerning the plants that were clustered together, and saves them in separate files. One file for each cluster
    :param path_to_dictionary: Path to the dictionary having as keys the cluster IDs and as values the list of IDs of
    the plants that were clustered together
    :param data_path: Folder containing one csv file for each plant
    :param dst_folder_name: Name of the folder where the file containing grouped plants will be saved
    :param axis: can be 0 or 1. If it is equal to 1, the dataframes will be concatenated horizontally. If it is 0, they
    will be concatenated vertically. The former case applies when we are preparing data for the multitarget model, that
    requires the series coming from the same plant cluster to be concatenated horizontally. If instead we are preparing
    data for training a single target model with series coming from all the plants in the cluster, they will be stacked
    vertically
    :param cols: List of the columns
    """
    #cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    #cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    os.makedirs(dst_folder, exist_ok=True)
    clusters_dict = load_from_pickle(path_to_dictionary)
    for k in clusters_dict.keys():
        df = pd.DataFrame()
        for id in clusters_dict[k]:
            if os.path.exists(data_path + "/" + id + ".csv"):
                d = pd.read_csv(data_path + "/" + id + ".csv")
                d = d[cols]
                df = pd.concat([df, d], ignore_index=True, axis=axis)
        df.to_csv(dst_folder + "/" + str(k) + ".csv")

def main(args):
    path_to_dictionary = args.path_to_dictionary
    train_data_path = args.train_data_path
    testing_data_path = args.testing_data_path
    number_of_columns = args.number_of_columns
    dst_folder = args.dst_folder
    axis = args.axis

    model_type = args.model_type
    n_clusters = path_to_dictionary.split('/')[-1].split('.')[0].split('_')[-1]
    columns = [str(c) for c in np.arange(number_of_columns)]
    c = path_to_dictionary.split("/")[-1].split("_")[0]
    dst_folder_train = "{}/{}/{}_{}/train".format(dst_folder, n_clusters, model_type, c)
    dst_folder_test = "{}/{}/{}_{}/test".format(dst_folder, n_clusters, model_type, c)
    chain(path_to_dictionary, train_data_path, dst_folder_train, axis, columns)
    chain(path_to_dictionary, testing_data_path, dst_folder_test, axis, columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dictionary", type=str, required=True, help="Path to the clustering dictionary")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the single target train data directory")
    parser.add_argument("--testing_data_path", type=str, required=True, help="Path to the single target test data directory")
    parser.add_argument("--dst_folder", type=str, required=True, help="Path to the directory where the aggregated datasets will be saved")
    parser.add_argument("--number_of_columns", type=int, required=False, help="Number of columns in the dataset")
    parser.add_argument("--axis", type=int, choices=[0, 1], help="Set this to 0 for vertically concatenating the rows. Set this to 1 for horizontally concatenating the rows.")
    parser.add_argument("--model_type", type=str, choices=["space", "time", "merged"], help="Either space (if the model will only use spatial clustering), "
                                                                                            "time (if the model will only use temporal clustering) or "
                                                                                            "merged if the model will used the merged clustering")
    args = parser.parse_args()
    main(args)
    # python chain_datasets.py --path_to_dictionary ../../clustering/spatial_clustering/clusters_dict_6.pkl --data_path ../pvitaly/single_target/test --dst_folder ../pvitaly/multitarget_space/6/test --axis 1
