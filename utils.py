import pickle
from sklearn.preprocessing import MinMaxScaler

def save_to_pickle(name, c):
    with open(name, 'wb') as f:
        pickle.dump(c, f)

def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def create_lstm_tensors(df, scaler, target_column):
    months_columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']     # Names of the columns containing the training data
    x = df[months_columns].values
    y = df[target_column].values

    x = scaler.fit_transform(x)
    x = x.reshape(x.shape[0], x.shape[1], 1)

    y = y.reshape(y.shape[0], 1)
    y = scaler.fit_transform(y)     # If we are testing we don't need to scale the y values

    return x, y


