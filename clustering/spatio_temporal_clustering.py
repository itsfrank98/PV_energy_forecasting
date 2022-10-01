import os
from utils import load_from_pickle, save_to_pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


fig, axs = plt.subplots(3, 2)
def perform_clustering(adj_matrix, title, x, y):
    nu = [2,3,4,5,6,7,8,9,10]
    sil = []
    for n in nu:
        cl = AgglomerativeClustering(n, affinity='precomputed', linkage='single')
        cl.fit(adj_matrix)
        sil.append(silhouette_score(adj_matrix, cl.labels_))
    #figure = plt.figure(1)
    #ax = figure.add_subplot(111)
    axs[x, y].plot(nu, sil, color='b', marker='.')
    axs[x, y].set_title(title)
    plt.ylabel("Inertia")
    plt.xlabel("K")

dir = 'spatio_temporal_clustering/matrices/mod_complete'
x = 0
y = 0
for f in os.listdir(dir):
    mat = load_from_pickle(os.path.join(dir, f))
    perform_clustering(mat, f.split(".")[0], x, y)
    if x == 0:
        if y == 0:
            y = 1
        elif y == 1:
            x = 1
            y = 0
    elif x == 1:
        if y == 0:
            y = 1
        elif y == 1:
            x = 2
            y = 0

for ax in axs.flat:
    ax.set(xlabel='n_clusters', ylabel='silhouette')
fig.tight_layout(pad=0.5)
plt.show()
