import arff
import numpy as np
import math
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.elbow import elbow
import matplotlib.pyplot as plt
from utils import save_to_pickle, load_from_pickle

def create_distance_matrix(dst_path, src='Fumagalli 8fold CV/train_2019.arff'):
    d = {}
    for row in arff.load(src):
        index = row[0]
        if index not in d.keys():
            lat = row[1]
            long = row[2]
            d[index] = (lat, long)

    keys_list = list(d.keys())
    distance_mat = np.zeros((len(d.keys()), len(d.keys())))
    for i in range(len(d.keys())):
        for j in range(len(d.keys())):
            if i != j:
                distance_mat[i, j] = math.dist(d[keys_list[i]], d[keys_list[j]])
    #save_to_pickle('spatial_clustering/plant_coordinates_dictionary.pkl', d)
    #save_to_pickle(dst_path, distance_mat)

# Plot inertia for each k in order to find the best number of clusters with the elbow method
def plot_inertia(n_clusters, inertias):
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.plot(range(1, len(n_clusters)+1), inertias, color='b', marker='.')
    ax.grid(True)
    plt.ylabel("Silhouette")
    plt.xlabel("K")
    plt.show()

'''def pyclustering_clustering():
    kmin, kmax = 1, 10
    elb = elbow(distance_mat, kmin, kmax)
    elb.process()
    amount = elb.get_amount()
    wce = elb.get_wce()
    #amount = 8
    initial_medoids = kmeans_plusplus_initializer(np.asmatrix(distance_mat), amount).initialize(return_index=True)
    clustering2 = kmedoids(np.asmatrix(distance_mat), initial_medoids, data_type='distance_matrix')
    clustering2.process()
    clusters = clustering2.get_clusters()

    elbow_instance = elbow(distance_mat, kmin, kmax, initializer=initial_medoids)
    elbow_instance.process()
    print(amount)
    print(wce)
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.plot(range(0, kmax), wce, color='b', marker='.')
    ax.plot(amount, wce[amount - kmin], color='r', marker='.', markersize=10)
    ax.annotate("Elbow", (amount + 0.1, wce[amount - kmin] + 5))
    ax.grid(True)
    plt.ylabel("WCE")
    plt.xlabel("K")
    plt.show()'''


def main(n_clusters, distance_mat_path, linkage="average", load=True):
    if not load:
        create_distance_matrix(distance_mat_path, '../Fumagalli 8fold CV/train_2019.arff')
    distance_mat = load_from_pickle(distance_mat_path)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=linkage)
    #clustering = KMedoids(n_clusters=n_clusters, random_state=0, metric='precomputed')
    clustering.fit(distance_mat)

    lab = list(clustering.labels_)
    labels_set = set(lab)
    plant_coordinates_dict = load_from_pickle("spatial_clustering/plant_coordinates_dictionary")
    ids = list(plant_coordinates_dict.keys())
    clusters_dict = {}
    for label in labels_set:
        d = []
        for i in range(len(lab)):
            if lab[i] == label:
                d.append(ids[i])
        clusters_dict[label] = d

    save_to_pickle("spatial_clustering/{}.pkl".format(n_clusters), clustering)

    print(d)

#n/2
main(n_clusters=75, distance_mat_path="spatial_clustering/distance_mat.pkl", load=True)
