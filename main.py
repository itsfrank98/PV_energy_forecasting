import sys
sys.path.append("model/prepare_data/")
sys.path.append("model/")
from clustering.merge_clusterings import main as merged_main
from clustering.spatio_temporal_clustering import main as spatio_temporal_main
from model.aggregated_model import main as aggregated_model_main
from model.prepare_data.aggregated import main as aggregated_main
import argparse

dataset = "pvitaly"
dst = "clustering/merged_clustering/{}/matrices".format(dataset)
train_data_path = "{}/single_target/train".format(dataset)
test_data_path = "{}/single_target/test".format(dataset)
def merge_clusters(s, t, technique):
    merged_parse = argparse.Namespace
    merged_parse.spatial_path = "{}/clustering/spatial_clustering/{}.pkl".format(dataset, s)
    merged_parse.temporal_path = "{}/clustering/temporal_clustering/{}.pkl".format(dataset, t)
    merged_parse.dst = dst
    merged_parse.spatial_dist_mat = "{}/clustering/spatial_clustering/distance_mat.pkl".format(dataset)
    merged_parse.temporal_dist_mat = "{}/clustering/temporal_clustering/distance_matrix.pkl".format(dataset)
    merged_parse.technique = technique
    merged_main(merged_parse)

def spatio_temporal_clustering(s, t, technique, n):
    spatio_temp_parse = argparse.Namespace
    spatio_temp_parse.n_clusters = n
    spatio_temp_parse.distance_mat_path = "{}/{}_{}s{}t.pkl".format(dst, technique, s, t)
    path_to_dictionary = "clustering/merged_clustering/{}/{}/{}s{}t_dict_{}.pkl".format(dataset, technique, s, t, n)
    spatio_temp_parse.clusters_dict_path = path_to_dictionary
    spatio_temp_parse.clustering_model_path = "clustering/merged_clustering/{}/{}/{}s{}t_clustering_{}.pkl".format(dataset, technique, s, t, n)
    spatio_temp_parse.dataset = dataset
    spatio_temporal_main(spatio_temp_parse)
    return path_to_dictionary

def aggregate(path_to_dictionary, technique):
    aggregated_parse = argparse.Namespace
    aggregated_parse.train_data_path = train_data_path
    aggregated_parse.testing_data_path = test_data_path
    aggregated_parse.s = "merged"
    aggregated_parse.path_to_dictionary = path_to_dictionary
    aggregated_parse.number_of_columns = 18
    aggregated_parse.dst_folder = "model/{}_merged_{}/".format(dataset, technique)
    aggregated_parse.y_cols = ['17']
    aggregated_main(aggregated_parse)

def single_target_clustering():


def train_test_model(path_to_dictionary, technique, n):
    c = path_to_dictionary.split("/")[-1].split("_")[0]
    model_parse = argparse.Namespace
    model_parse.train_dir = "model/{}_merged_{}/{}/aggregated_{}/train".format(dataset, technique, n, c)
    model_parse.test_dir = "model/{}_merged_{}/{}/aggregated_{}/test".format(dataset, technique, n, c)
    model_parse.file_name = "model/{}_merged_{}/{}/aggregated_{}/results.txt".format(dataset, technique, n, c)
    model_parse.model_folder = "model/{}_merged_{}/{}/aggregated_{}/models".format(dataset, technique, n, c)
    model_parse.neurons = 18
    model_parse.dropout = 0.3
    model_parse.lr = 0.008
    model_parse.batch_size = 200
    model_parse.epochs = 200
    model_parse.patience = 10
    model_parse.preprocess = 0
    model_parse.clustering_dict_path = path_to_dictionary
    model_parse.y_column = 1
    aggregated_model_main(model_parse)


if __name__ == "__main__":
    for technique in ["modularity", "consensus", "silhouette", "mi"]:
        for tup in [(4, 5), (5, 4), (4, 4), (5, 5)]:
            s = tup[0]
            t = tup[1]
            merge_clusters(s, t, technique)
            for n in [4,5,6]:
                path_to_dictionary = spatio_temporal_clustering(s, t, technique, n)
                aggregate(path_to_dictionary, technique)
                train_test_model(path_to_dictionary, technique, n)
