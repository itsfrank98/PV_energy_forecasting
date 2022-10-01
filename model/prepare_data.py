import arff
import pandas as pd
import os

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


if __name__ == "__main__":
    prepare_data_single_dataset("../Fumagalli 8fold CV/train_2019.arff", "single_datasets/train")
    prepare_data_single_dataset("../Fumagalli 8fold CV/test_2019.arff", "single_datasets/test")
