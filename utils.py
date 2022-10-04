import pickle

import pandas as pd
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

def create_lstm_tensors(df, target_column, max=False):
    months_columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']     # Names of the columns containing the training data
    x = df[months_columns].values
    y = df[target_column].values
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
    '''x = scaler.fit_transform(x)
    
    y = scaler.fit_transform(y)'''
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



