import sys
sys.path.append("model/prepare_data/")
sys.path.append("model/")
from clustering.merge_clusterings import main as merged_main
from clustering.spatio_temporal_clustering import main as spatio_temporal_main
from model.aggregated_model import main as aggregated_model_main
from model.run_models import main as single_target_clustering_model_main
from model.prepare_data.aggregate import main as aggregated_main
from model.prepare_data.multitarget_prepare import main as prepare_single_target_clustering_main
import argparse

dataset = "pvitaly"
dst = "clustering/merged_clustering/{}/matrices".format(dataset)
train_data_path = "{}/single_target/train".format(dataset)
test_data_path = "{}/single_target/test".format(dataset)
def merge_clusters(s, t, technique):
    parser = argparse.Namespace
    parser.spatial_path = "{}/clustering/spatial_clustering/{}.pkl".format(dataset, s)
    parser.temporal_path = "{}/clustering/temporal_clustering/{}.pkl".format(dataset, t)
    parser.dst = dst
    parser.spatial_dist_mat = "{}/clustering/spatial_clustering/distance_mat.pkl".format(dataset)
    parser.temporal_dist_mat = "{}/clustering/temporal_clustering/distance_matrix.pkl".format(dataset)
    parser.technique = technique
    merged_main(parser)

def spatio_temporal_clustering(s, t, technique, n, path_to_dictionary):
    parser = argparse.Namespace
    parser.n_clusters = n
    parser.distance_mat_path = "{}/{}_{}s{}t.pkl".format(dst, technique, s, t)
    parser.clusters_dict_path = path_to_dictionary
    parser.clustering_model_path = "clustering/merged_clustering/{}/{}/{}s{}t_clustering_{}.pkl".format(dataset, technique, s, t, n)
    parser.dataset = dataset
    spatio_temporal_main(parser)

def prepare_data(path_to_dictionary, technique, aggregate, model_type):
    parser = argparse.Namespace
    parser.train_data_path = train_data_path
    parser.testing_data_path = test_data_path
    parser.path_to_dictionary = path_to_dictionary
    parser.dst_folder = "model/{}_merged_{}".format(dataset, technique)
    parser.number_of_columns = 18
    if aggregate:
        parser.s = "merged"
        parser.y_cols = ['17']
        aggregated_main(parser)
    else:
        parser.axis = 0
        parser.model_type = model_type
        prepare_single_target_clustering_main(parser)

def train_test_model(path_to_dictionary, technique, n, model_type):
    c = path_to_dictionary.split("/")[-1].split("_")[0]
    parser = argparse.Namespace
    parser.train_dir = "model/{}_merged_{}/{}/{}_{}/train".format(dataset, technique, n, model_type, c)
    parser.test_dir = "model/{}_merged_{}/{}/{}_{}/test".format(dataset, technique, n, model_type, c)
    parser.file_name = "model/{}_merged_{}/{}/{}_{}/results.txt".format(dataset, technique, n, model_type, c)
    parser.model_folder = "model/{}_merged_{}/{}/{}_{}/models".format(dataset, technique, n, model_type, c)
    parser.neurons = 18
    parser.dropout = 0.3
    parser.lr = 0.005
    parser.batch_size = 200
    parser.epochs = 1
    parser.patience = 10
    parser.preprocess = 0
    parser.clustering_dict_path = path_to_dictionary
    parser.y_column = -1
    if model_type == "aggregated":
        aggregated_model_main(parser)
    elif model_type == "single_model_clustering":
        parser.training_type = model_type
        single_target_clustering_model_main(parser)


if __name__ == "__main__":
    # for model_type in ['aggregated', 'single_model_clustering']:
    model_type = 'single_model_clustering' #
    for technique in ["consensus", "modularity",  "silhouette", "mi"]:
        for tup in [(4, 5), (5, 4), (4, 4), (5, 5)]:
            s = tup[0]
            t = tup[1]
            merge_clusters(s, t, technique)
            for n in [4, 5, 6]:
                path_to_dictionary = "clustering/merged_clustering/{}/{}/{}s{}t_dict_{}.pkl".format(dataset, technique, s, t, n)
                spatio_temporal_clustering(s=s, t=t, technique=technique, n=n, path_to_dictionary=path_to_dictionary)
                aggregate = False
                if model_type == 'aggregated':
                    aggregate = True
                prepare_data(path_to_dictionary=path_to_dictionary, technique=technique, aggregate=aggregate, model_type=model_type)
                train_test_model(path_to_dictionary=path_to_dictionary, technique=technique, n=n, model_type=model_type)
