import numpy as np
from sklearn.metrics import silhouette_samples
import argparse
import sys
sys.path.append("..")
from utils import load_from_pickle, save_to_pickle


def point_consensus(spatial_labels, temporal_labels):
    """
    This is the naive technique. It directly computes the final matrix by following a simple procedure. Consider the points (i, j). In the final matrix,
    at the position [i, j], we will put:
    - 2 if they were clustered together by both the clustering models
    - 1 if they were clustered together by only one of the clustering models
    - 0 if they were never clustered together
    :param spatial_labels: Labels that the spatial clustering model assigned to the points
    :param temporal_labels: Labels that the temporal clustering model assigned to the points
    :return:
    """
    consensus_matrix = np.ones((len(spatial_labels), len(spatial_labels))) * 2      # Values on diagonal will be always equal to 2
    for i in range(len(spatial_labels)):
        for j in range(i+1, len(spatial_labels)):
            value = 0
            if (spatial_labels[i] == spatial_labels[j]) != (temporal_labels[i] == temporal_labels[j]):  # Simulation of XOR. Case in which the two points are clustered together by only one of the clustering models
                value = 1
            elif spatial_labels[i] == spatial_labels[j] and temporal_labels[i] == temporal_labels[j]:
                value = 2
            consensus_matrix[i, j] = value
            consensus_matrix[j, i] = value
    return consensus_matrix

def compute_cluster_modularity(labels):
    # Calculates the modularity between clusters
    d = {l: np.count_nonzero(labels == l) for l in set(labels)}     # Dictionary having as keys the cluster ids and as values the number of elements in each cluster
    cluster_modularity_matrix = np.zeros((len(set(labels)), len(set(labels))))
    keys = list(d.keys())
    for i in range(len(keys)):
        elements_in_i = d[keys[i]]
        for j in range(len(keys)):
            elements_in_j = d[keys[j]]
            g = elements_in_i + elements_in_j     # Total number of elements in the two clusters
            m = g*(g-1)     # How many arcs we would have if all the elements we are considering were fully connected
            lj = elements_in_j*(elements_in_j-1)/2  # Number of arcs in the j community if they were fully connected
            kj = elements_in_j*(elements_in_j-1)    # Sum of the grades of the nodes in the j community if they were fully connected
            modularity = (lj/m)-(kj/(2*m))**2
            cluster_modularity_matrix[i, j] = modularity
    return cluster_modularity_matrix

def compute_modularity_adj_matrix(labels):
    # Create an adjacency matrix that in the position [i, j] has the modularity between the clusters to which the points i and j belong
    N = len(labels)
    modularity_matrix = compute_cluster_modularity(labels)
    modularity_adj_matrix = np.zeros((N, N))
    for i in range(N):
        c_i = labels[i]
        for j in range(N):
            c_j = labels[j]
            modularity_adj_matrix[i, j] = modularity_matrix[c_i, c_j]
    return modularity_adj_matrix

def compute_silhouette_adj_matrix(adj_matrix, labels):
    """
    Computes the silhouette score between points. The final adjacency matrix, at the position [i, j], will contain the product between the silhouettes
    obtained for the points (i, j)
    :param adj_matrix: Adjacency matrix used for creating the clustering model.
    :param labels: Labels that the clustering model assigned to the points
    :return:
    """
    silhouette_values = silhouette_samples(adj_matrix, labels)
    silhouette_adj_matrix = np.zeros((len(labels), len(labels)))
    for i in range(len(labels)):
        s_i = silhouette_values[i]
        for j in range(len(labels)):
            s_j = silhouette_values[j]
            silhouette_adj_matrix[i, j] = s_i * s_j
    return silhouette_adj_matrix

def dictionary_indexes(labels):
    """
    Create a dictionary having as keys the IDs of the clusters and as values the list of the indexes of the elements
    belonging to the correspondant cluster. For instance suppose to have this list of labels: [0,2,1,1,0], the dictionary
    will be:
    {
        0: [0, 4],
        1: [2, 3],
        2: [1]
    }
    This dictionary will be used to find the intersection between two clusters produced by different models
    """
    d = {}
    for l in set(labels):
        d[l] = list(np.where(np.array(labels) == l))
    return d

def cluster_intersection_matrix(labels1, labels2):
    """
    Supposing to have two clustering models C1 and C2, create a table that at position [i, j] has the number of elements
    in the intersection between the i-th cluster in C1 and the j-th cluster in C2.
    :param labels1: Labels obtained by the first clustering model
    :param labels2: Labels obtained by the second clustering model
    :return:
    """
    d1 = dictionary_indexes(labels1)
    d2 = dictionary_indexes(labels2)
    max = np.max((len(set(labels1)), len(set(labels2))))
    intersection_matrix = np.zeros((max, max))
    for k1 in d1.keys():
        for k2 in d2.keys():
            intersection_matrix[k1, k2] = len(np.intersect1d(d1[k1], d2[k2]))
    return intersection_matrix

def compute_mi_matrix(spatial_labels, temporal_labels):
    """
    Computes an adjacency matrix having as value the mutual information between points
    :param spatial_labels: Labels that the spatial clustering model computed for the points
    :param temporal_labels: Labels that the temporal clustering model computed for the points
    :return:
    """
    N = len(spatial_labels)
    d_prob_spatial = {l: np.count_nonzero(spatial_labels==l)/len(spatial_labels) for l in set(spatial_labels)}   # Dictionary that associates at each cluster the probability that a point has to belong to that cluster
    d_prob_temporal = {l: np.count_nonzero(temporal_labels==l)/len(temporal_labels) for l in set(temporal_labels)}
    intersection_matrix = cluster_intersection_matrix(spatial_labels, temporal_labels)

    mi_matrix = np.zeros((N, N))
    for i in range(len(spatial_labels)):
        lab_i = spatial_labels[i]
        P_i = d_prob_spatial[lab_i]
        for j in range(len(temporal_labels)):
            lab_j = temporal_labels[j]
            P_j = d_prob_temporal[lab_j]
            P_ij = intersection_matrix[lab_i, lab_j]/N
            if P_ij != 0:
                mi_matrix[i, j] = P_ij * np.log((P_ij/(P_i*P_j)))
    return mi_matrix


def main(args):
    spatial_clustering_path = args.spatial_path
    temp_clustering_path = args.temporal_path
    n_clusters_space = spatial_clustering_path.split("/")[3].split(".")[0]
    n_clusters_time = temp_clustering_path.split("/")[3].split(".")[0]
    technique = args.technique
    dst = args.dst
    spatial_clustering = load_from_pickle(spatial_clustering_path)
    temporal_clustering = load_from_pickle(temp_clustering_path)

    spatial_labels = spatial_clustering.labels_
    temporal_labels = temporal_clustering.labels_

    if technique == "consensus":
        adj_matrix = point_consensus(spatial_labels, temporal_labels)
    elif technique == "modularity":
        adj_matrix_spatial = compute_modularity_adj_matrix(spatial_labels)
        adj_matrix_temporal = compute_modularity_adj_matrix(temporal_labels)
        adj_matrix = adj_matrix_spatial + adj_matrix_temporal
    elif technique == "silhouette":
        dist_matrix_spatial = load_from_pickle(args.spatial_dist_mat)
        dist_matrix_temporal = load_from_pickle(args.temporal_dist_mat)
        silhouette_matrix_spatial = compute_silhouette_adj_matrix(dist_matrix_spatial, spatial_labels)
        silhouette_matrix_temporal = compute_silhouette_adj_matrix(dist_matrix_temporal, temporal_labels)
        adj_matrix = silhouette_matrix_spatial + silhouette_matrix_temporal
    elif technique == "mi":
        adj_matrix = compute_mi_matrix(spatial_labels, temporal_labels)
    else:
        raise ValueError("ERROR: Invalid technique")
    save_to_pickle("{}/{}_{}s{}t.pkl".format(dst, technique, n_clusters_space, n_clusters_time), adj_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial_path", type=str, required=True, help="Path to the pickle file containing the spatial clustering")
    parser.add_argument("--temporal_path", type=str, required=True, help="Path to the pickle file containing the temporal clustering")
    parser.add_argument("--technique", type=str, required=True, help="Technique to use to merge the clusterings. Can be consensus, modularity, mi, silhouette",
                        choices=["consensus", "modularity", "mi", "silhouette"])
    parser.add_argument("--spatial_dist_mat", type=str, required=False, help="Path to the file containing the spatial clustering adjacency matrix. This parameter is used only if the technique is 'silhouette'")
    parser.add_argument("--temporal_dist_mat", type=str, required=False, help="Path to the file containing the temporal clustering adjacency matrix. This parameter is used only if the technique is 'silhouette'")
    parser.add_argument("--dst", type=str, required=True)
    args = parser.parse_args()
    #main(args)

