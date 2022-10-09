import sys
sys.path.append('../')

import arff
import pandas as pd
import os
from utils import load_from_pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse

def prepare_data_single_dataset(data_path, dst_folder):
    """
    This function takes the original dataset and splits it in multiple ones. More precisely, it creates a separated
    dataset for each plant. These datasets will be used to train the single target models related to each plant.
    Then, they will be merged to train the multitarget models
    :return:
    """
    if not os.path.isdir(os.path.join(dst_folder)):
        os.mkdir(os.path.join(dst_folder))
    current_id = ""
    x_positions = [7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40]   # List of indexes containing, in the row, the relevant data
    l_series = []
    for row in arff.load(data_path):
        row = list(row)
        id = row[0]
        if current_id != id:
            df = pd.DataFrame(l_series)
            df.to_csv(dst_folder+"/{}.csv".format(current_id))
            current_id = id
            l_series = []
        l = []
        for i in x_positions:
            l.append(row[i])
        l.append(row[-1])
        l_series.append(l)

def create_data_multi_target_dataset(path_to_dictionary, dataset_folder, dst_folder_path):
    """
    Groups together the series concerning the plants that were clustered together, and saves them in separate files
    :param path_to_dictionary: Path to the dictionary having as keys the cluster IDs and as values the list of IDs of
    the plants that were clustered together
    :param dataset_folder: Folder containing one csv file for each plant
    :param dst_folder_name: Name of the folder where the file containing grouped plants will be saved
    :return:
    """
    os.makedirs(dst_folder_path, exist_ok=True)
    clusters_dict = load_from_pickle(path_to_dictionary)
    for k in clusters_dict.keys():
        df = pd.DataFrame()
        file_name = ""
        for id in clusters_dict[k]:
            if id != "133.0":   # For some reason, the csv corresponding to the plant with id '133.0' wasn't created so we skip this id
                d = pd.read_csv(dataset_folder+"/"+id+".csv")
                df = pd.concat([df, d], ignore_index=True)
                file_name += id.split('.')[0]+"_"
        df.to_csv(dst_folder_path+"/"+file_name[:-1]+".csv")       # We use file_name[:-1] to prevent it from ending in "_", which isn't aesthetic


def absolute_scaler(x, max, testing=False):
    if testing:
        for i in range(len(x)):
            if i <= max:
                x[i] /= max
            else:
                x[i] = 1
    else:
        x = x/max
    return x

def create_lstm_tensors(df, max=False):
    # USELESS
    data = df.values
    x, y = data[:, 1:-1], data[:, -1]
    testing = True
    if not max:
        max_x = np.max(x)
        max_y = np.max(y)
        if max_x > max_y:
            max = max_x
        else:
            max = max_y
        testing=False

    x = absolute_scaler(x, max, testing=testing)
    y = absolute_scaler(y, max, testing=testing)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = y.reshape(y.shape[0], 1)

    return x, y, max

def create_lstm_tensors_minmax(df, scaler):
    data = df.values
    if not scaler:
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(data)
    else:
        scaled = scaler.transform(data)

    x, y = scaled[:, :-1], scaled[:, -1]
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = y.reshape(y.shape[0], 1)

    return x, y, scaler

def main(args):
    type = args.type
    if type == "single_target":
        data_path = args.data_path
        dst_folder = args.dst
        prepare_data_single_dataset(data_path=data_path, dst_folder=dst_folder)
    elif type == "multi_target":
        path_to_dictionary = args.dict_src
        dataset_folder = args.dataset_folder
        dst_folder_path = args.dst
        create_data_multi_target_dataset(path_to_dictionary=path_to_dictionary, dataset_folder=dataset_folder, dst_folder_path=dst_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, help="Type of dataset", choices=['single_target', 'multi_target'])
    parser.add_argument("--dst", type=str, required=True, help="Destination directory")
    parser.add_argument("--data_path", type=str, required=False, help="Path to the data used to create the single target dataset")
    parser.add_argument("--dict_src", type=str, required=False, help="Path to the clustering dictionary")
    parser.add_argument("--dataset_folder", type=str, required=False, help="Path to the folder containing the dataset")

    args = parser.parse_args()
    main(args)

#python prepare_data.py --type multi_target --dict_src ../clustering/spatial_clustering/clusters_dict_40.pkl --dataset_folder single_datasets/test --dst multitarget_40_space/test
