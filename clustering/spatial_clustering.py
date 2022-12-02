import os
import sys
sys.path.append('../')
import arff
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
#import matplotlib.pyplot as plt
import pandas as pd
from utils import save_to_pickle, load_from_pickle, create_clusters_dict
import argparse
from tqdm import tqdm

def create_dict_from_arff(src, dictionary=None):
    d = {}
    for row in arff.load(src):
        index = row[0]
        if dictionary:
            if dictionary[index] < 24:
                continue
        if index not in d.keys():
            lat = row[1]
            lon = row[2]
            d[index] = np.array((lat, lon))
    return d

def create_dict_pvitaly(src="../datasets/pvitaly/splitted"):
    d = {}
    for f in os.listdir(src):
        df = pd.read_csv(os.path.join(src, f))[['lat', 'lon']]
        index = f[:-4]
        lat = df.loc[0].lat
        lon = df.loc[0].lon
        d[index] = np.array(lat, lon)
    return d

def create_distance_matrix(d, dst_dir):
    keys_list = list(d.keys())
    distance_mat = np.zeros((len(d.keys()), len(d.keys())))
    for i in tqdm(range(len(d.keys()))):
        for j in range(len(d.keys())):
            if i != j:
                distance_mat[i, j] = np.linalg.norm(d[keys_list[i]] - d[keys_list[j]])
    save_to_pickle('{}/plant_coordinates_dictionary.pkl'.format(dst_dir), d)
    save_to_pickle('{}/distance_matrix.pkl'.format(dst_dir), distance_mat)


def main(args):
    n_clusters = args.n_clusters
    dst_directory = args.dst_directory
    linkage = args.linkage
    load = args.load
    dataset = args.dataset

    if not load:
        if dataset in ["latiano", "foggia"]:
            if dataset == "latiano":
                dictionary = None
                src = "../datasets/Fumagalli 8fold CV/train_2019.arff"
            else:
                dictionary = load_from_pickle("../datasets/SS-DT_foggia_prod/dict.pkl")
                src = "../datasets/SS-DT_foggia_prod/train_2019.arff"
            d = create_dict_from_arff(src=src, dictionary=dictionary)
        elif dataset == "pvitaly":
            d = create_dict_pvitaly('../datasets/pvitaly/splitted')
        else:
            raise ValueError("You didn't specify a dataset")
        create_distance_matrix(d, dst_directory)
    distance_mat = load_from_pickle('{}/distance_matrix.pkl'.format(dst_directory))
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage)
    #clustering = KMedoids(n_clusters=n_clusters, random_state=0, metric='precomputed')
    clustering.fit(distance_mat)

    lab = list(clustering.labels_)
    plant_coordinates_dict = load_from_pickle("{}/plant_coordinates_dictionary.pkl".format(dst_directory))
    ids = list(plant_coordinates_dict.keys())
    clusters_dict = create_clusters_dict(lab, ids)

    save_to_pickle("{}/{}.pkl".format(dst_directory, n_clusters), clustering)
    save_to_pickle("{}/clusters_dictionary_{}.pkl".format(dst_directory, n_clusters), clusters_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, required=True, help="How many clusters to create")
    parser.add_argument("--dst_directory", type=str, required=True, help="Directory where the files will be saved")
    parser.add_argument("--linkage", type=str, required=True, help="Linkage type")  #average
    parser.add_argument("--load", type=bool, required=False, help="Whether to load the distance matrix or create a new one")
    parser.add_argument("--dataset", type=str, required=False, help="Name of the dataset on which clustering will be performed. It can either be 'latiano' or 'pvitaly'."
                                                                    "If you already have the distance matrix and don't want it to be computed again, you can ignore this argument",
                        choices=['latiano', 'pvitaly', 'foggia'])

    args = parser.parse_args()
    main(args)

# python spatial_clustering.py --n_clusters 4 --distance_mat_path spatial_clustering/distance_mat.pkl --clusters_dict_name spatial_clustering/clusters_dict_4.pkl --linkage average --dataset pvitaly --load True
