from utils import load_from_pickle, save_to_pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def perform_clustering(adj_matrix):
    nu = [4,5,6,7,8,9,10]
    sil = []
    for n in nu:
        #cl = AgglomerativeClustering(n, affinity='precomputed', linkage='average')
        cl =KMedoids(n_clusters=n, metric="precomputed")
        cl.fit(adj_matrix)
        sil.append(silhouette_score(adj_matrix, cl.labels_))
    figure = plt.figure(1)
    ax = figure.add_subplot(111)
    ax.plot(nu, sil, color='b', marker='.')
    ax.grid(True)
    plt.ylabel("Inertia")
    plt.xlabel("K")
    plt.show()

mat = load_from_pickle('spatio_temporal_clustering/matrices/mi/mi_96months.pkl')
perform_clustering(mat)
