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

def create_latiano_dictionary(n):
    d = {}
    current_index = ""
    for row in arff.load("../datasets/Fumagalli 8fold CV/train_2019.arff"):
        index = row[0]
        if index != current_index:
            d[index] = []  # List where we put all the energy values registered by a plant each month from 2011 to 2018
            current_index = index
        d[index].append(row[41])
    save_to_pickle("../latiano/clustering/dictionary_dtw_{}.pkl".format(n), d)

def create_dictionary_dtw_pvitaly(v):
    """
    Creates the DTW dictionary for the pvitaly dataset. The dictionary has as keys the plant ids and as value a list of the
    kwh values to consider when computing the DTW distance
    :param v: How many values to consider to compute the DTW distance. The v most recent values will be taken
    """
    src = "../datasets/pvitaly/splitted"
    d = {}
    for f in os.listdir(src):
        index = f[:-4]
        df = pd.read_csv(os.path.join(src, f))[['kwh']]
        d[index] = list(df.kwh)[-v:]
    save_to_pickle("dictionary_dtw_{}.pkl".format(v), d)


def create_distance_matrix(d, dst):
    distance_matrix = np.zeros((len(d.keys()), len(d.keys())))
    for i in tqdm(range(len(d.keys()))):
        k1 = list(d.keys())[i]
        for j in range(i+1, len(d.keys())):
            k2 = list(d.keys())[j]
            dtw_dist = dtw.distance(d[k1], d[k2])
            distance_matrix[i, j] = dtw_dist
            distance_matrix[j, i] = dtw_dist
    save_to_pickle(dst, distance_matrix)

def main(args):
    n_clusters = args.n_clusters
    distance_mat_path = args.distance_mat_path
    clusters_dict_path = args.clusters_dict_path
    linkage = args.linkage
    v = args.v
    dataset = args.dataset

    if dataset == "latiano":
        create_latiano_dictionary(v)
    elif dataset == "pvitaly":
        create_dictionary_dtw_pvitaly(v)
    dictionary_dtw = load_from_pickle("../latiano/clustering/dictionary_dtw_{}.pkl".format(v))
    create_distance_matrix(dictionary_dtw, distance_mat_path)
    distance_mat = load_from_pickle(distance_mat_path)
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage=linkage)
    #clustering_model = KMedoids(n_clusters=n_clusters, random_state=0, metric='precomputed')

    clustering_model.fit(distance_mat)
    lab = list(clustering_model.labels_)
    ids = list(dictionary_dtw.keys())
    clusters_dict = create_clusters_dict(lab, ids)

    save_to_pickle("temporal_clustering/{}.pkl".format(n_clusters), clustering_model)
    save_to_pickle(clusters_dict_path, clusters_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, required=True, help="How many clusters to create")
    parser.add_argument("--distance_mat_path", type=str, required=False, help="Path where the distance matrix will be saved")
    parser.add_argument("--clusters_dict_path", type=str, required=True, help="Path where the clusters' dictionary will be saved")
    parser.add_argument("--linkage", type=str, required=True, help="Linkage type")  #average
    parser.add_argument("--v", type=int, required=False, help="How many values to consider for the time series.")
    parser.add_argument("--dataset", type=str, required=False, help="Name of the dataset on which clustering ill be performed. It can either be 'latiano' or 'pvitaly'.",
                        choices=['latiano', 'pvitaly'])
    args = parser.parse_args()
    main(args)
    '''d = load_from_pickle("dictionary_dtw.pkl")
    create_distance_matrix(d)'''
# python temporal_clustering.py --n_clusters 6 --distance_mat_path temporal_clustering/dtw_matrices/distance_matrix_38.pkl --clusters_dict_path temporal_clustering/clusters_dict_6.pkl --linkage average --v 38 --dataset pvitaly
