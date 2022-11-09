import pickle
import os

def save_to_pickle(name, c):
    with open(name, 'wb') as f:
        pickle.dump(c, f)

def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def create_clusters_dict(labels, plants_ids):
    '''
    Create a dictionary having as keys the clusters' ids and as values the list of IDs of the plants belonging to each cluster
    Suppose to have two clusters and 5 plants, the dictionary is:
    {0: ['1.0', '4.0'], 1: ['0.0', '2.0', '3.0']}
    :param labels: The list of labels that the clustering model assigned to each plant
    :param plants_ids: The list of IDs of the plants
    :return:
    '''
    labels_set = set(labels)
    clusters_dict = {}
    for label in labels_set:
        d = []
        for i in range(len(labels)):
            if labels[i] == label:
                d.append(plants_ids[i])
        clusters_dict[label] = d
    return clusters_dict

def sort_results(unsorted_file_path, sorted_file_path):
    dic = {}
    s1 = 0
    s2 = 0
    with open(unsorted_file_path, 'r') as f:
        for line in f:
            d = line.split()
            k = d[0][:-1]
            if k != "MA":
                #a = float(d[1])
                #s1 += a
                #dic[k] = a
                mae, rmse = float(d[1]), float(d[2])
                s1 += mae
                s2 += rmse
                dic[k] = (mae, rmse)
        f.close()
    with open(sorted_file_path, 'w') as f:
        f.write("      MAE         RMSE\n")
        for k in sorted(dic.keys()):
            f.write("{}:  {}\n".format(k, dic[k]))
        f.write("AVG:  {}  {}".format(s1/len(dic.keys()), s2/len(dic.keys())))
        #f.write("AVG:  {}".format(s1/len(dic.keys())))
        f.close()
    #os.remove(unsorted_file_path)
