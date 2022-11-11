import argparse
import sys

import numpy as np

sys.path.append('../..')
import os
from tqdm import tqdm
import pandas as pd
from multitarget_prepare import prepare_data_multitarget
from utils import load_from_pickle
def prepare_data_aggregated_train(data_path, path_to_dictionary, dst_folder, number_of_cols):
    """
    Create dataframes for another type of training: the one where we have 24 features for each plant. The first 12 are
    the energy values for that plant, the other 12 are obtained by aggregating the features of the other plants in the
    same cluster as the one we are considering. They are aggregated by computing the mean.
    :return:
    """
    cols = [str(c) for c in range(number_of_cols)]
    prepare_data_multitarget(path_to_dictionary, data_path, dst_folder, axis=0)
    for f in tqdm(os.listdir(os.path.join(dst_folder))):
        d = pd.read_csv(os.path.join(dst_folder, f))
        d = d.loc[:, d.columns != 'Unnamed: 0']
        means = d[cols].mean(axis=0)
        means = means.values
        means = np.expand_dims(means, axis=0)
        means = np.repeat(means, len(d), axis=0)
        df_means = pd.DataFrame(data=means, index=None, columns=[str(i)+'m' for i in range(number_of_cols)])
        #print(df_means)
        definitive_df = pd.concat((df_means.loc[:, df_means.columns != 'index'], d), axis=1)
        definitive_df.to_csv(os.path.join(dst_folder, f))

def prepare_data_aggregated_test(clustering_dict, dst_folder, train_dir, number_of_cols, testing_data_path):
    for f in tqdm(os.listdir(testing_data_path)):
        if f == ".csv":
            continue
        fname = f.split(".")[0]
        test_df = pd.read_csv(os.path.join(testing_data_path, fname+".0.csv"))
        mean_cols = [str(c)+'m' for c in np.arange(number_of_cols)]

        # Retrieve the cluster to which the plant belongs
        for k in clustering_dict.keys():
            if fname+".0" in clustering_dict[k]:
                cluster = str(k)
                break
        df_means = pd.read_csv(os.path.join(train_dir, cluster+".csv"))[mean_cols][:len(test_df)]
        definitive_df = pd.concat((df_means.loc[:, df_means.columns != 'index'], test_df), axis=1)
        definitive_df = definitive_df.loc[:, definitive_df.columns != 'Unnamed: 0']
        definitive_df.to_csv(os.path.join(dst_folder, f))

def main(args):
    train_data_path = args.train_data_path
    testing_data_path = args.testing_data_path
    space = args.space
    path_to_dictionary = args.path_to_dictionary
    clustering_dict = load_from_pickle(path_to_dictionary)
    n_of_cols = args.number_of_columns
    n_clusters = path_to_dictionary.split('/')[-1].split('.')[0].split('_')[-1]

    if space == True:
        a = "space"
    else:
        a = "time"
    dst_folder_train = "../pvitaly/multitarget_vertical/{}/{}/train".format(a, n_clusters)
    dst_folder_test = "../pvitaly/multitarget_vertical/{}/{}/test".format(a, n_clusters)
    os.makedirs(dst_folder_train, exist_ok=True)
    os.makedirs(dst_folder_test, exist_ok=True)
    prepare_data_aggregated_train(data_path=train_data_path, path_to_dictionary=path_to_dictionary, dst_folder=dst_folder_train,
                                  number_of_cols=n_of_cols)
    prepare_data_aggregated_test(clustering_dict=clustering_dict, train_dir=dst_folder_train, dst_folder=dst_folder_test,
                                 number_of_cols=n_of_cols, testing_data_path=testing_data_path)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the single target train data directory")
    parser.add_argument("--testing_data_path", type=str, required=True, help="Path to the single target test data directory")
    parser.add_argument("--space", type=bool, required=False, help="Set to true if it is spatial clustering. Set to False or ignore if it is temporal clustering")
    parser.add_argument("--path_to_dictionary", type=str, required=True, help="Path to the clustering dictionary")
    parser.add_argument("--number_of_columns", type=int, required=False, help="Number of columns in the dataset")

    args = parser.parse_args()
    main(args)
#python aggregated.py --train_data_path ../single_target/train/ --space True --path_to_dictionary ../../clustering/spatial_clustering/clusters_dict_10_aggl.pkl
