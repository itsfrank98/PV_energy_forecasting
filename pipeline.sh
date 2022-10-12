cd clustering/
python spatial_clustering.py --n_clusters 10 --distance_mat_path spatial_clustering/distance_mat.pkl --clusters_dict_name spatial_clustering/clusters_dict_10.pkl --linkage average --load True
cd ../model/
python prepare_data.py --type multi_target --dict_src ../clustering/spatial_clustering/clusters_dict_10.pkl --dataset_folder single_datasets/test --dst multitarget_10_space/test
python prepare_data.py --type multi_target --dict_src ../clustering/spatial_clustering/clusters_dict_10.pkl --dataset_folder single_datasets/train --dst multitarget_10_space/train
python models.py --train_dir multitarget_10_space/train --test_dir multitarget_10_space/test --file_name multitarget_10_space/results.txt --neurons 12 --dropout 0.3 --lr 0.005 --model_folder multitarget_10_space/models --model_type multi_target --epochs 200
