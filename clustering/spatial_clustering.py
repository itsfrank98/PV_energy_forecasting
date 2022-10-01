import arff
import numpy as np
import math
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.elbow import elbow
import matplotlib.pyplot as plt
from utils import save_to_pickle

d = {}
for row in arff.load('Fumagalli 8fold CV/train_2019.arff'):
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

n_clusters = 5
inertias = []
#clustering = KMedoids(n_clusters=n_clusters, random_state=0, metric='precomputed').fit(distance_mat)
n_clusters = np.arange(3, 15)
for n in n_clusters:
    clustering = AgglomerativeClustering(n_clusters=n, affinity="precomputed", linkage="average")
    #clustering = KMedoids(n_clusters=n, random_state=0, metric='precomputed')
    clustering.fit(distance_mat)
    lab = list(clustering.labels_)
    if n == 8:
        save_to_pickle("{}.pkl".format(n), clustering)
        d = {l+1: lab.count(l) for l in set(lab)}
        print(d)
    inertias.append(silhouette_score(distance_mat, lab))


#save_to_pickle("spatial_adj_matrix.pkl", distance_mat)
#save_to_pickle("{}.pkl".format(n_clusters), clustering)
#inertias.append(clustering.inertia_)

# Plot inertia for each k in order to find the best number of clusters with the elbow method
def plot_inertia(n_clusters, inertias):
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.plot(range(1, len(n_clusters)+1), inertias, color='b', marker='.')
    ax.grid(True)
    plt.ylabel("Silhouette")
    plt.xlabel("K")
    plt.show()
plot_inertia(n_clusters, inertias)
def pyclustering_clustering():
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
    plt.show()


