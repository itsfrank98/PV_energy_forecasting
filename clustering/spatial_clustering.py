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

def create_dict_latiano(src='datasets/Fumagalli 8fold CV/train_2019.arff'):
    d = {}
    for row in arff.load(src):
        index = row[0]
        if index not in d.keys():
            lat = row[1]
            lon = row[2]
            d[index] = np.array(lat, lon)
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

def create_distance_matrix(d, dst_path):
    keys_list = list(d.keys())
    distance_mat = np.zeros((len(d.keys()), len(d.keys())))
    for i in range(len(d.keys())):
        for j in range(len(d.keys())):
            if i != j:
                distance_mat[i, j] = np.linalg.norm(d[keys_list[i]] - d[keys_list[j]])
    save_to_pickle('spatial_clustering/plant_coordinates_dictionary.pkl', d)
    save_to_pickle(dst_path, distance_mat)


# Plot inertia for each k in order to find the best number of clusters with the elbow method
'''def plot_inertia(n_clusters, inertias):
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.plot(range(1, len(n_clusters)+1), inertias, color='b', marker='.')
    ax.grid(True)
    plt.ylabel("Silhouette")
    plt.xlabel("K")
    plt.show()'''

def main(args):
    n_clusters = args.n_clusters
    distance_mat_path = args.distance_mat_path
    clusters_dict_name = args.clusters_dict_name
    linkage = args.linkage
    load = args.load
    dataset = args.dataset

    if not load:
        if dataset == "latiano":
            d = create_dict_latiano('../datasets/Fumagalli 8fold CV/train_2019.arff')
        elif dataset == "pvitaly":
            d = create_dict_pvitaly('../datasets/pvitaly/splitted')
        else:
            raise ValueError("You didn't specify a dataset")
        create_distance_matrix(d, distance_mat_path)
    distance_mat = load_from_pickle(distance_mat_path)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage)
    #clustering = KMedoids(n_clusters=n_clusters, random_state=0, metric='precomputed')
    clustering.fit(distance_mat)

    lab = list(clustering.labels_)
    plant_coordinates_dict = load_from_pickle("spatial_clustering/plant_coordinates_dictionary.pkl")
    ids = list(plant_coordinates_dict.keys())
    clusters_dict = create_clusters_dict(lab, ids)

    save_to_pickle("spatial_clustering/{}.pkl".format(n_clusters), clustering)
    save_to_pickle(clusters_dict_name, clusters_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, required=True, help="How many clusters to create")
    parser.add_argument("--distance_mat_path", type=str, required=False, help="Path to the distance matrix")
    parser.add_argument("--clusters_dict_name", type=str, required=True, help="How the clusters' dictionary will be saved")
    parser.add_argument("--linkage", type=str, required=True, help="Linkage type")  #average
    parser.add_argument("--load", type=bool, required=False, help="Whether to load the distance matrix or create a new one")
    parser.add_argument("--dataset", type=str, required=False, help="Name of the dataset on which clustering ill be performed. It can either be 'latiano' or 'pvitaly'."
                                                                    "If you already have the distance matrix and don't want it to be computed again, you can ignore this argument",
                        choices=['latiano', 'pvitaly'])

    args = parser.parse_args()
    main(args)

# python spatial_clustering.py --n_clusters 40 --distance_mat_path spatial_clustering/distance_mat.pkl --clusters_dict_name spatial_clustering/clusters_dict_40.pkl --linkage average --load True
