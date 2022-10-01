import arff
from dtaidistance import dtw
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
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
linkage="average"      # Average linkage was found to be the best one among the four alternatives
matrices_directory = "dtw_matrices"
d_months2clusters = {   # This dictionary stores the number of clusters to create depending on how many months we consider
    '6': 8,
    '12': 6,
    '36': 5,
    '72': 6,
    '96': 6
}
for f in os.listdir(matrices_directory):
    number_of_months = f.split('_')[-1].split('.')[0]
    with open(os.path.join(matrices_directory, f), 'rb') as f:
        distance_mat = pickle.load(f)
        n_clusters = d_months2clusters[number_of_months]
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage=linkage)
        clustering_model.fit(distance_mat)
        pickle.dump(clustering_model, open("temporal_clustering/{}months_{}.pkl".format(number_of_months, n_clusters), 'wb'))
