import arff
from dtaidistance import dtw
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def create_dictionary():
    d = {}
    current_index = ""
    for row in arff.load("Fumagalli 8fold CV/train_2019.arff"):
        index = row[0]
        if index != current_index:
            d[index] = []  # List where we put all the energy values registered by a plant  each month from 2011 to 2018
            current_index = index
        d[index].append(row[41])
    pickle.dump(d, open("dictionary_dtw.pkl", 'wb'))

# create_dictionary()

def create_distance_matrix(d):
    values = [6, 12, 36, 72, 96]
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
        pickle.dump(distance_matrix, open("distance_matrix_{}.pkl".format(v), 'wb'))


with open("dictionary_dtw.pkl", 'rb') as f:
    d = pickle.load(f)
# create_distance_matrix(d)

with open("dtw_matrices/distance_matrix_96.pkl", 'rb') as f:
    distance_mat = pickle.load(f)

n_clusters = np.arange(15,30)
inertias = []
for n in n_clusters:
    clustering = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage="single").fit(distance_mat)
    lab = list(clustering.labels_)
    d_foo ={l+1: lab.count(l) for l in set(lab)}
    inertias.append(clustering.)

figure = plt.figure(1)
ax = figure.add_subplot(111)
ax.plot(n_clusters, inertias, color='b', marker='.')
ax.grid(True)
plt.ylabel("Inertia")
plt.xlabel("K")
plt.show()
