import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
from prepare_data.single_dataset import prepare_data_single_dataset
from prepare_data.multitarget_modeling import prepare_data_multitarget


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
        prepare_data_multitarget(path_to_dictionary=path_to_dictionary, data_path=dataset_folder, dst_folder=dst_folder, axis=axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True, help="Type of dataset", choices=['single_target', 'multi_target', 'single_target_clustering'])
    parser.add_argument("--dst", type=str, required=True, help="Destination directory")
    parser.add_argument("--data_path", type=str, required=False, help="Path to the data used to create the single target dataset")
    parser.add_argument("--dict_src", type=str, required=False, help="Path to the clustering dictionary")
    parser.add_argument("--dataset_folder", type=str, required=False, help="Path to the folder containing the dataset")

    args = parser.parse_args()
    main(args)

#python preprocess.py --type multi_target --dict_src ../clustering/spatial_clustering/clusters_dict_40.pkl --dataset_folder single_datasets/test --dst multitarget_40_space/test
#python preprocess.py --type single_target --data_path ../Fumagalli\ 8fold\ CV/test_2019.arff --dst single_datasets/6months/test
