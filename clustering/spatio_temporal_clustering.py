from sklearn.cluster import AgglomerativeClustering
import argparse
import sys
sys.path.append("../")
from utils import load_from_pickle, save_to_pickle, create_clusters_dict

def main(args):
    n_clusters = args.n_clusters
    distance_mat_path = args.distance_mat_path
    clusters_dict_path = args.clusters_dict_path
    clustering_model_path = args.clustering_model_path
    dataset = args.dataset

    distance_mat = load_from_pickle(distance_mat_path)
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage="average")
    #clustering_model = KMedoids(n_clusters=n_clusters, random_state=0, metric='precomputed')
    clustering_model.fit(distance_mat)

    plant_coordinates_dict = load_from_pickle("{}/clustering/spatial_clustering/plant_coordinates_dictionary.pkl".format(dataset))
    ids = list(plant_coordinates_dict.keys())
    lab = list(clustering_model.labels_)
    clusters_dict = create_clusters_dict(lab, ids)

    save_to_pickle(clusters_dict_path, clusters_dict)
    save_to_pickle(clustering_model_path, clustering_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clusters", type=int, required=True, help="How many clusters to create")
    parser.add_argument("--distance_mat_path", type=str, required=False, help="Path where the distance matrix will be saved")
    parser.add_argument("--clusters_dict_path", type=str, required=True, help="Path where the clusters' dictionary will be saved")
    parser.add_argument("--clustering_model_path", type=str, required=True, help="Path where the clustering model will be saved")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset on which clustering will be performed. It can either be 'latiano' or 'pvitaly'.",
                        choices=['latiano', 'pvitaly'])
    args = parser.parse_args()
    main(args)
