import os
import pandas as pd
import arff
import argparse
import sys
sys.path.append('../..')
from utils import load_from_pickle

def prepare_data_single_dataset(data_path, dst_folder, target_index, dict=None):
    """
    This function takes the original dataset and splits it in multiple ones. More precisely, it creates a separated
    dataset for each plant.
    :return:
    """
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if not os.path.isdir(os.path.join(dst_folder)):
        os.mkdir(os.path.join(dst_folder))
    current_id = ""
    #x_positions = [7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40]   # List of indexes containing, in the row, the relevant data
    x_positions = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    l_series = []
    for row in arff.load(data_path):
        row = list(row)
        id = row[0]
        if dict:
            if id in dict.keys and dict[id] < 24:
                continue
        if current_id == "":
            current_id = id
        if current_id != id and l_series:
            df = pd.DataFrame(l_series)
            df = df[cols]
            df.to_csv(dst_folder+"/{}.csv".format(current_id))
            current_id = id
            l_series = []
        l = []
        for i in x_positions:
            l.append(row[i])
        l.append(row[target_index])
        l_series.append(l)

def main(args):
    # type = args.type
    dst_folder = args.dst
    data_path = args.data_path
    dataset = args.dataset

    os.makedirs(dst_folder, exist_ok=True)
    dictionary = None
    if dataset == "fumagalli":
        target_index = -1
    elif dataset in ["foggia_train", "foggia_test"]:
        target_index = 3
        if dataset == "foggia_train":
            dictionary = load_from_pickle("../../datasets/SS-DT_foggia_prod/dict.pkl")
    prepare_data_single_dataset(data_path=data_path, dst_folder=dst_folder, target_index=target_index, dict=dictionary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--type", type=str, required=True, help="Type of dataset", choices=['single_target', 'multi_target', 'single_target_clustering'])
    parser.add_argument("--dst", type=str, required=True, help="Destination directory")
    parser.add_argument("--data_path", type=str, required=False, help="Path to the data used to create the single target dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name", choices=["foggia_train", "foggia_test", "fumagalli"])
    args = parser.parse_args()
    main(args)
