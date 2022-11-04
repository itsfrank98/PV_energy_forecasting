import argparse
import sys
sys.path.append('../..')
import os
from tqdm import tqdm
import pandas as pd
from multitarget_modeling import prepare_data_multitarget
from utils import load_from_pickle
def prepare_data_aggregated_train(data_path, path_to_dictionary, dst_folder):
    """
    Create dataframes for another type of training: the one where we have 24 features for each plant. The first 12 are
    the energy values for that plant, the other 12 are obtained by aggregating the features of the other plants in the
    same cluster as the one we are considering. They are aggregated by computing the mean.
    :return:
    """
    cols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    prepare_data_multitarget(path_to_dictionary, data_path, dst_folder, axis=0)
    for f in tqdm(os.listdir(os.path.join(dst_folder))):
        d = pd.read_csv(os.path.join(dst_folder, f))
        d = d.loc[:, d.columns != 'Unnamed: 0']
        means = d[cols].mean(axis=0)
        df1 = pd.DataFrame(means)
        df_means = pd.DataFrame()
        for _ in range(len(d)):
            df_means = pd.concat((df_means, df1), axis=1)
        df_means = df_means.T.reset_index()
        df_means = df_means.rename(columns={'0': '0m', '1': '1m', '2': '2m', '3': '3m', '4': '4m', '5': '5m', '6': '6m', '7': '7m', '8': '8m', '9': '9m', '10': '10m', '11': '11m'})

        definitive_df = pd.concat((df_means.loc[:, df_means.columns != 'index'], d), axis=1)
        definitive_df.to_csv(os.path.join(dst_folder, f))

def prepare_data_aggregated_test(clustering_dict, dst_folder, train_dir):
    p = "../single_target/test"
    for f in tqdm(os.listdir(p)):
        if f == ".csv":
            continue
        fname = f.split(".")[0]
        test_df = pd.read_csv(os.path.join(p, fname+".0.csv"))
        mean_cols = ['0m', '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m', '10m', '11m']

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
    space = args.space
    path_to_dictionary = args.path_to_dictionary
    clustering_dict = load_from_pickle(path_to_dictionary)

    n_clusters = path_to_dictionary.split('/')[-1].split('.')[0].split('_')[-1]
    if space:
        a = "space"
    else:
        a = "time"
    dst_folder_train = "../multitarget_vertical/{}/kmed/{}/train".format(a, n_clusters)
    dst_folder_test = "../multitarget_vertical/{}/kmed/{}/test".format(a, n_clusters)
    os.makedirs(dst_folder_train, exist_ok=True)
    os.makedirs(dst_folder_test, exist_ok=True)
    prepare_data_aggregated_train(data_path=train_data_path, path_to_dictionary=path_to_dictionary, dst_folder=dst_folder_train)
    prepare_data_aggregated_test(clustering_dict=clustering_dict, train_dir=dst_folder_train, dst_folder=dst_folder_test)


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the single target train data directory")
    parser.add_argument("--space", type=bool, required=False, help="Set to true if it is spatial clustering. Set to False or ignore if it is temporal clustering")
    parser.add_argument("--path_to_dictionary", type=str, required=True, help="Path to the clustering dictionary")
    args = parser.parse_args()
    main(args)
#python aggregated.py --train_data_path ../single_target/train/ --space True --path_to_dictionary ../../clustering/spatial_clustering/clusters_dict_10_aggl.pkl
