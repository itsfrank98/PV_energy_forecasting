import argparse
import sys
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from multitarget_prepare import prepare_data_multitarget
sys.path.append('../..')
from utils import load_from_pickle
def prepare_data_aggregated_train(data_path, path_to_dictionary, dst_folder, x_cols, y_cols):
    """
    Create dataframes for another type of training: the one where we have 24 features for each plant. The first 12 are
    the energy values for that plant, the other 12 are obtained by aggregating the features of the other plants in the
    same cluster as the one we are considering. They are aggregated by computing the mean.
    :return:
    """

    prepare_data_multitarget(path_to_dictionary, data_path, dst_folder, axis=0, cols=x_cols+y_cols)
    for f in tqdm(sorted(os.listdir(os.path.join(dst_folder)))):
        if f == "18.csv":  #37
            continue
        d = pd.read_csv(os.path.join(dst_folder, f))
        d = d.loc[:, d.columns != 'Unnamed: 0']
        means = d[x_cols].mean(axis=0)
        means = means.values
        means = np.expand_dims(means, axis=0)
        means = np.repeat(means, len(d), axis=0)
        df_means = pd.DataFrame(data=means, index=None, columns=[str(i)+'m' for i in range(len(x_cols))])
        #print(df_means)
        definitive_df = pd.concat((df_means.loc[:, df_means.columns != 'index'], d), axis=1)
        definitive_df.to_csv(os.path.join(dst_folder, f))

def prepare_data_aggregated_test(clustering_dict, dst_folder, train_dir, x_cols, testing_data_path):
    for f in tqdm(os.listdir(testing_data_path)):
        if f == ".csv":
            continue
        fname = f.split(".")[0]
        test_df = pd.read_csv(os.path.join(testing_data_path, fname+".0.csv"))
        mean_cols = [str(c)+'m' for c in x_cols]

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
    s = args.s
    path_to_dictionary = args.path_to_dictionary
    clustering_dict = load_from_pickle(path_to_dictionary)
    n_of_cols = args.number_of_columns
    n_clusters = path_to_dictionary.split('/')[-1].split('.')[0].split('_')[-1]
    dst_folder = args.dst_folder
    y_cols = args.y_cols

    if s == "merged":
        c = path_to_dictionary.split("/")[-1].split("_")[0]
        dst_folder_train = "{}/{}/aggregated_{}/train".format(dst_folder, n_clusters, c)
        dst_folder_test = "{}/{}/aggregated_{}/test".format(dst_folder, n_clusters, c)
    else:
        dst_folder_train = "{}/{}/{}/train".format(dst_folder, s, n_clusters)
        dst_folder_test = "{}/{}/{}/test".format(dst_folder, s, n_clusters)

    os.makedirs(dst_folder_train, exist_ok=True)
    os.makedirs(dst_folder_test, exist_ok=True)

    cols = [str(c) for c in range(n_of_cols)]
    x_cols = [c for c in cols if c not in y_cols]
    prepare_data_aggregated_train(data_path=train_data_path, path_to_dictionary=path_to_dictionary, dst_folder=dst_folder_train,
                                  x_cols=x_cols, y_cols=y_cols)
    prepare_data_aggregated_test(clustering_dict=clustering_dict, train_dir=dst_folder_train, dst_folder=dst_folder_test,
                                 testing_data_path=testing_data_path, x_cols=x_cols)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the single target train data directory")
    parser.add_argument("--testing_data_path", type=str, required=True, help="Path to the single target test data directory")
    parser.add_argument("--s", type=str, required=True, help="type 'space' if you're considering spatial clustering, 'time' if temporal clustering, 'merged' if both", choices=['space', 'time', 'merged'])
    parser.add_argument("--path_to_dictionary", type=str, required=True, help="Path to the clustering dictionary")
    parser.add_argument("--number_of_columns", type=int, required=False, help="Number of columns in the dataset")
    parser.add_argument("--dst_folder", type=str, required=True, help="Folder where the created datasets will be put")
    parser.add_argument("--y_cols", nargs='+', required=True, help="Index(es) of the column(s) with the target feature. Aggregation won't be performed on this column")

    args = parser.parse_args()
    main(args)
#python aggregate.py --train_data_path ../single_target/train/ --s time --path_to_dictionary ../../clustering/spatial_clustering/clusters_dict_10_aggl.pkl
