import sys
sys.path.append('../')
import arff
from dtaidistance import dtw
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
import pickle
from tqdm import tqdm
from utils import save_to_pickle, load_from_pickle
from spatial_clustering import create_clusters_dict
import argparse

def create_dictionary():
    d = {}
    current_index = ""
    for row in arff.load("Fumagalli 8fold CV/train_2019.arff"):
        index = row[0]
        if index != current_index:
            d[index] = []  # List where we put all the energy values registered by a plant each month from 2011 to 2018
            current_index = index
        d[index].append(row[41])
    pickle.dump(d, open("dictionary_dtw.pkl", 'wb'))


def create_distance_matrix(d, dst="clustering/dtw_matrices"):
    values = [2, 4]
    n_of_temp = 12*8
    for v in values:
        distance_matrix = np.zeros((len(d.keys()), len(d.keys())))
        for i in tqdm(range(len(d.keys()))):
            k1 = list(d.keys())[i]
            for j in range(i+1, len(d.keys())):
                k2 = list(d.keys())[j]
                dtw_dist = dtw.distance(d[k1][n_of_temp-v:], d[k2][n_of_temp-v:])
                distance_matrix[i, j] = dtw_dist
                distance_matrix[j, i] = dtw_dist
        pickle.dump(distance_matrix, open("{}/distance_matrix_{}.pkl".format(dst, v), 'wb'))

def main(args):
    n_clusters = args.n_clusters
    distance_mat_path = args.distance_mat_path
    clusters_dict_name = args.clusters_dict_name
    linkage = args.linkage

    distance_mat = load_from_pickle(distance_mat_path)
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage=linkage)
    #clustering_model = KMedoids(n_clusters=n_clusters, random_state=0, metric='precomputed')

    clustering_model.fit(distance_mat)

    dictionary_dtw = load_from_pickle("dictionary_dtw.pkl")
    lab = list(clustering_model.labels_)
    ids = list(dictionary_dtw.keys())
    clusters_dict = create_clusters_dict(lab, ids)

    save_to_pickle("temporal_clustering/{}".format(n_clusters), clustering_model)
    save_to_pickle(clusters_dict_name, clusters_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, required=True, help="How many clusters to create")
    parser.add_argument("--distance_mat_path", type=str, required=False, help="Path to the distance matrix")
    parser.add_argument("--clusters_dict_name", type=str, required=True, help="How the clusters' dictionary will be saved")
    parser.add_argument("--linkage", type=str, required=True, help="Linkage type")  #average
    args = parser.parse_args()
    main(args)
    '''d = load_from_pickle("dictionary_dtw.pkl")
    create_distance_matrix(d)'''
# python temporal_clustering.py --n_clusters 40 --distance_mat_path dtw_matrices/distance_matrix_96.pkl --clusters_dict_name temporal_clustering/clusters_dict_40.pkl --linkage average

