import sys
sys.path.append('../')
import arff
import pandas as pd
import os
from utils import load_from_pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
from tqdm import tqdm

def prepare_data_single_dataset(data_path, dst_folder):
    """
    This function takes the original dataset and splits it in multiple ones. More precisely, it creates a separated
    dataset for each plant. These datasets will be used to train1 the single target models related to each plant.
    Then, they will be merged to train the multitarget models
    :return:
    """
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if not os.path.isdir(os.path.join(dst_folder)):
        os.mkdir(os.path.join(dst_folder))
    current_id = ""
    #7, 10, 13, 16, 19, 22,
    x_positions = [25, 28, 31, 34, 37, 40, 7, 10, 13, 16, 19, 22]   # List of indexes containing, in the row, the relevant data
    l_series = []
    for row in arff.load(data_path):
        row = list(row)
        id = row[0]
        if current_id != id and l_series:
            df = pd.DataFrame(l_series)
            df = df[cols]
            df.to_csv(dst_folder+"/{}.csv".format(current_id))
            current_id = id
            l_series = []
        l = []
        for i in x_positions:
            l.append(row[i])
        l.append(row[-1])
        l_series.append(l)


def create_data_multi_dataset(path_to_dictionary, data_path, dst_folder, axis):
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
    cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    os.makedirs(dst_folder, exist_ok=True)
    clusters_dict = load_from_pickle(path_to_dictionary)
    for k in clusters_dict.keys():
        df = pd.DataFrame()
        for id in clusters_dict[k]:
            if id != "133.0":   # For some reason, the csv corresponding to the plant with id '133.0' wasn't created so we skip this id
                d = pd.read_csv(data_path + "/" + id + ".csv")
                d = d[cols]
                df = pd.concat([df, d], ignore_index=True, axis=axis)
        '''with open(dst_folder_path + "/ids.txt", 'w') as f:
            f.write(file_name)
        file_name = "tutti_"'''
        df.to_csv(dst_folder + "/" + str(k) + ".csv")


def create_lstm_tensors_minmax(df, scaler, aggregate_training):
    data = df.values
    columns = np.arange(start=1, stop=len(df.columns))
    y_columns = np.arange(start=12, stop=len(df.columns)+1, step=13) + 1    # Every 12 columns we have the value of the target variable
    if aggregate_training:
        y_columns = [25]
    print(data[:, y_columns])
    x_columns = [c for c in columns if c not in y_columns]
    if not scaler:
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(data)
    else:
        scaled = scaler.transform(data)
    x, y = scaled[:, x_columns], scaled[:, y_columns]
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y_flat = []
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y_flat.append(y[i, j])
    y_flat = np.array(y_flat)
    y_flat = y_flat.reshape(y_flat.shape[0], 1)

    # If we are testing we need to deal with the out of range values
    if scaler:
        x[x<0] = 0
        x[x>1] = 1
        y_flat[y_flat<0] = 0
        y_flat[y_flat>1] = 1
    return x, y_flat, scaler


def main(args):
    type = args.type
    dst_folder = args.dst
    os.makedirs(dst_folder, exist_ok=True)
    if type == "single_target":
        data_path = args.data_path
        prepare_data_single_dataset(data_path=data_path, dst_folder=dst_folder)
    else:
        path_to_dictionary = args.dict_src
        dataset_folder = args.dataset_folder
        if type == "multi_target":
            axis = 1
        elif type == "single_target_clustering":
            axis = 0
        create_data_multi_dataset(path_to_dictionary=path_to_dictionary, data_path=dataset_folder, dst_folder=dst_folder, axis=axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, help="Type of dataset", choices=['single_target', 'multi_target', 'single_target_clustering'])
    parser.add_argument("--dst", type=str, required=True, help="Destination directory")
    parser.add_argument("--data_path", type=str, required=False, help="Path to the data used to create the single target dataset")
    parser.add_argument("--dict_src", type=str, required=False, help="Path to the clustering dictionary")
    parser.add_argument("--dataset_folder", type=str, required=False, help="Path to the folder containing the dataset")

    args = parser.parse_args()
    main(args)

#python prepare_data.py --type multi_target --dict_src ../clustering/spatial_clustering/clusters_dict_40.pkl --dataset_folder single_datasets/test --dst multitarget_40_space/test
#python prepare_data.py --type single_target --data_path ../Fumagalli\ 8fold\ CV/test_2019.arff --dst single_datasets/6months/test
