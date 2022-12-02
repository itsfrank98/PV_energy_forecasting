import os
import sys
import pandas as pd
sys.path.append('../')
import arff
from dtaidistance import dtw
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from utils import save_to_pickle, load_from_pickle, create_clusters_dict
import argparse

def create_dictionary_dtw_from_arff(src_path, l, target_index, dst, dictionary=None):
    """
    Function that creates the DTW dictionary by extracting information from an arff file. The dictionary has as keys the
    plant ids and as value a list of the :n: kwh values to consider for computing the DTW distance
    :param src_path: Path to the source arff file
    :param n: Number of elements to consider
    :param target_index: Position of the target value inside each row
    :param dst: Path to the directory where the dtw dictionary will be saved
    :return:
    """
    d = {}
    current_index = ""
    for row in arff.load(src_path):
        index = row[0]
        if dictionary:
            if dictionary[index] < 24:
                continue
        if index != current_index:
            if current_index != "":
                d[current_index] = d[current_index][-l:]  # When a new index is met the list associated to the previous index is cut to the last n values
            d[index] = []  # List where we put all the energy values registered by a plant each month from 2011 to 2018
            current_index = index
        d[index].append(row[target_index])
    save_to_pickle("{}/dictionary_dtw_{}.pkl".format(dst, l), d)

def create_dictionary_dtw_pvitaly(l, dst):
    """
    Creates the DTW dictionary for the pvitaly dataset. The dictionary has as keys the plant ids and as value a list of the
    kwh values to consider when computing the DTW distance
    :param l: How many values to consider to compute the DTW distance. The v most recent values will be taken
    """
    src = "../datasets/pvitaly/splitted"
    d = {}
    for f in os.listdir(src):
        index = f[:-4]
        df = pd.read_csv(os.path.join(src, f))[['kwh']]
        d[index] = list(df.kwh)[-l:]
    save_to_pickle("{}/dictionary_dtw_{}.pkl".format(dst, l), d)


def create_distance_matrix(d, dst):
    timeseries = []
    #distance_matrix = np.zeros((len(d.keys()), len(d.keys())))
    keys_list = list(d.keys())
    for i in range(len(d.keys())):
        timeseries.append(np.array(d[keys_list[i]]))
    distance_matrix = dtw.distance_matrix_fast(timeseries)
    """for j in range(i+1, len(d.keys())):
        k2 = list(d.keys())[j]
        dtw_dist = dtw.distance(d[k1], d[k2])
        distance_matrix[i, j] = dtw_dist
        distance_matrix[j, i] = dtw_dist"""
    save_to_pickle(dst, distance_matrix)

def main(args):
    n_clusters = args.n_clusters
    dst_directory = args.dst_directory
    l = args.l
    dataset = args.dataset
    load = args.load

    if not load:
        os.makedirs(dst_directory, exist_ok=True)
        if dataset == "latiano":
            create_dictionary_dtw_from_arff(src_path="../datasets/Fumagalli 8fold CV/train_2019.arff", l=l, target_index=41, dst=dst_directory)
        elif dataset == "foggia":
            dictionary = load_from_pickle("../datasets/SS-DT_foggia_prod/dict.pkl")
            create_dictionary_dtw_from_arff(src_path="../datasets/SS-DT_foggia_prod/train_2019.arff", l=l, target_index=3, dst=dst_directory, dictionary=dictionary)
        elif dataset == "pvitaly":
            create_dictionary_dtw_pvitaly(l=l, dst=dst_directory)
    dictionary_dtw = load_from_pickle("{}/dictionary_dtw_{}.pkl".format(dst_directory, l))
    create_distance_matrix(dictionary_dtw, "{}/distance_matrix_{}.pkl".format(dst_directory, l))
    distance_mat = load_from_pickle("{}/distance_matrix_{}.pkl".format(dst_directory, l))
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage="average")
    #clustering_model = KMedoids(n_clusters=n_clusters, random_state=0, metric='precomputed')

    clustering_model.fit(distance_mat)
    lab = list(clustering_model.labels_)
    ids = list(dictionary_dtw.keys())
    clusters_dict = create_clusters_dict(lab, ids)

    save_to_pickle("{}/{}.pkl".format(dst_directory, n_clusters), clustering_model)
    save_to_pickle("{}/clusters_dictionary_{}.pkl".format(dst_directory, n_clusters), clusters_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, required=True, help="How many clusters to create")
    parser.add_argument("--dst_directory", type=str, required=True, help="Directory where the files will be saved")
    parser.add_argument("--l", type=int, required=False, help="Length of the time series to consider.")
    parser.add_argument("--dataset", type=str, required=False, help="Name of the dataset on which clustering will be performed. It can either be 'latiano' or 'pvitaly'.",
                        choices=['latiano', 'pvitaly', 'foggia'])
    parser.add_argument("--load", type=bool, required=False, help="Whether to load the distance matrix or create a new one")
    args = parser.parse_args()
    main(args)
    '''d = load_from_pickle("dictionary_dtw.pkl")
    create_distance_matrix(d)'''
# python temporal_clustering.py --n_clusters 6 --distance_mat_path temporal_clustering/dtw_matrices/distance_matrix_38.pkl --clusters_dict_path temporal_clustering/clusters_dict_6.pkl --linkage average --v 38 --dataset pvitaly
