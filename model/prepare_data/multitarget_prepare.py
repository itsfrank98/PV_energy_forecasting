"""Prepare files for multitarget modelling"""
import argparse
import sys
sys.path.append('../..')
from utils import load_from_pickle
import os
import pandas as pd

def prepare_data_multitarget(path_to_dictionary, data_path, dst_folder, axis):
    """
    Groups together the series concerning the plants that were clustered together, and saves them in separate files
    :param path_to_dictionary: Path to the dictionary having as keys the cluster IDs and as values the list of IDs of
    the plants that were clustered together
    :param data_path: Folder containing one csv file for each plant
    :param dst_folder_name: Name of the folder where the file containing grouped plants will be saved
    :param axis: can be 0 or 1. If it is equal to 1, the dataframes will be concatenated horizontally. If it is 0, they
    will be concatenated vertically. The former case applies when we are preparing data for the multitarget model, that
    requires the series coming from the same plant cluster to be concatenated horizontally. If instead we are preparing
    data for training a single target model with series coming from all the plants in the cluster, they will be stacked
    vertically
    :return:
    """
    cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    os.makedirs(dst_folder, exist_ok=True)
    clusters_dict = load_from_pickle(path_to_dictionary)
    for k in clusters_dict.keys():
        df = pd.DataFrame()
        for id in clusters_dict[k]:
            if id != "133.0":   # For some reason, the csv corresponding to the plant with id '133.0' wasn't created so we skip this id
                d = pd.read_csv(data_path + "/" + id + ".csv")
                d = d[cols]
                df = pd.concat([df, d], ignore_index=True, axis=1)  #PROBABILMENTE AXIS VA LASCIATO A 1 E RIMOSSO DAI PARAMETRI, DATO CHE LA CONCATENAZIONE IN VERTICALE VIENE FATTA DALLA FUNZIONE train_unique_model
        df.to_csv(dst_folder + "/" + str(k) + ".csv")

def main(args):
    path_to_dictionary = args.path_to_dictionary
    data_path = args.data_path
    dst_folder = args.dst_folder
    axis = args.axis
    prepare_data_multitarget(path_to_dictionary, data_path, dst_folder, axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dictionary", type=str, required=True, help="Path to the clustering dictionary")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing data to be aggregated")
    parser.add_argument("--dst_folder", type=str, required=True, help="Path to the destination directory")
    parser.add_argument("--axis", type=int, choices=[0, 1], help="Set this to 0 for vertically concatenating the rows. Set this to 1 for horizontally concatenating the rows.")

    args = parser.parse_args()
    main(args)
    # python multitarget_prepare.py --path_to_dictionary ../../clustering/spatial_clustering/clusters_dict_6.pkl --data_path ../pvitaly/single_target/test --dst_folder ../pvitaly/multitarget_space/6/test --axis 1
