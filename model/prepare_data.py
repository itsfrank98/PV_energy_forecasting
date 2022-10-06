import arff
import pandas as pd
import os
from utils import load_from_pickle

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

def create_set(path_to_dictionary, dataset_folder, dst_folder_path):
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

create_set("../clustering/spatial_clustering/clusters_dict.pkl", "single_datasets/train", "bo/train")

if __name__ == "__main__":
    prepare_data_single_dataset("../Fumagalli 8fold CV/train_2019.arff", "single_datasets/train")
    prepare_data_single_dataset("../Fumagalli 8fold CV/test_2019.arff", "single_datasets/test")
