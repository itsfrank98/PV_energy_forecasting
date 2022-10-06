import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def save_to_pickle(name, c):
    with open(name, 'wb') as f:
        pickle.dump(c, f)

def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def absolute_scaler(x, max, testing=False):
    if testing:
        for i in range(len(x)):
            if i <= max:
                x[i] /= max
            else:
                x[i] = 1
    else:
        x = x/max
    return x

def create_lstm_tensors(df, max=False):
    # USELESS
    data = df.values
    x, y = data[:, 1:-1], data[:, -1]
    testing = True
    if not max:
        max_x = np.max(x)
        max_y = np.max(y)
        if max_x > max_y:
            max = max_x
        else:
            max = max_y
        testing=False

    x = absolute_scaler(x, max, testing=testing)
    y = absolute_scaler(y, max, testing=testing)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = y.reshape(y.shape[0], 1)

    return x, y, max

def create_lstm_tensors_minmax(df, scaler):
    data = df.values
    if not scaler:
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(data)
    else:
        scaled = scaler.transform(data)

    x, y = scaled[:, :-1], scaled[:, -1]
    x = x.reshape(x.shape[0], x.shape[1], 1)
    y = y.reshape(y.shape[0], 1)

    return x, y, scaler

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

