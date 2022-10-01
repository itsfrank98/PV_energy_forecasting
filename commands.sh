#!/bin/bash
python merge_clusterings.py --spatial_path 7complete.pkl --spatial_dist_mat spatial_adj_matrix.pkl --temporal_path temporal_clustering/clusters/96months_6.pkl --temporal_dist_mat dtw_matrices/distance_matrix_96.pkl --technique silhouette --dst spatio_temporal_clustering/matrices/sil_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --spatial_dist_mat spatial_adj_matrix.pkl --temporal_path temporal_clustering/clusters/36months_5.pkl --temporal_dist_mat dtw_matrices/distance_matrix_36.pkl --technique silhouette --dst spatio_temporal_clustering/matrices/sil_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --spatial_dist_mat spatial_adj_matrix.pkl --temporal_path temporal_clustering/clusters/6months_8.pkl --temporal_dist_mat dtw_matrices/distance_matrix_6.pkl --technique silhouette --dst spatio_temporal_clustering/matrices/sil_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --spatial_dist_mat spatial_adj_matrix.pkl --temporal_path temporal_clustering/clusters/12months_6.pkl --temporal_dist_mat dtw_matrices/distance_matrix_12.pkl --technique silhouette --dst spatio_temporal_clustering/matrices/sil_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --spatial_dist_mat spatial_adj_matrix.pkl --temporal_path temporal_clustering/clusters/72months_6.pkl --temporal_dist_mat dtw_matrices/distance_matrix_72.pkl --technique silhouette --dst spatio_temporal_clustering/matrices/sil_complete/

python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/6months_8.pkl --technique consensus --dst spatio_temporal_clustering/matrices/cons_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/12months_6.pkl --technique consensus --dst spatio_temporal_clustering/matrices/cons_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/36months_5.pkl --technique consensus --dst spatio_temporal_clustering/matrices/cons_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/72months_6.pkl --technique consensus --dst spatio_temporal_clustering/matrices/cons_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/96months_6.pkl --technique consensus --dst spatio_temporal_clustering/matrices/cons_complete/

python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/96months_6.pkl --technique modularity --dst spatio_temporal_clustering/matrices/mod_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/72months_6.pkl --technique modularity --dst spatio_temporal_clustering/matrices/mod_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/36months_5.pkl --technique modularity --dst spatio_temporal_clustering/matrices/mod_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/12months_6.pkl --technique modularity --dst spatio_temporal_clustering/matrices/mod_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/6months_8.pkl --technique modularity --dst spatio_temporal_clustering/matrices/mod_complete/

python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/96months_6.pkl --technique mi --dst spatio_temporal_clustering/matrices/mi_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/72months_6.pkl --technique mi --dst spatio_temporal_clustering/matrices/mi_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/36months_5.pkl --technique mi --dst spatio_temporal_clustering/matrices/mi_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/6months_8.pkl --technique mi --dst spatio_temporal_clustering/matrices/mi_complete/
python merge_clusterings.py --spatial_path 7complete.pkl --temporal_path temporal_clustering/clusters/12months_6.pkl --technique mi --dst spatio_temporal_clustering/matrices/mi_complete/

