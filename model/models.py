""" File containing the functions to create the single target and the multitarget models """
import keras
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense, Input
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import create_lstm_tensors
import pandas as pd
import tensorflow as tf
import os
import numpy as np
import random as rn
from tqdm import tqdm

np.random.seed(42)
tf.random.set_seed(42)
rn.seed(42)


def create_single_target_model(neurons, dropout, x_train, lr):
    i = Input(shape=(x_train.shape[1], x_train.shape[2]))
    n = LSTM(units=neurons)(i)
    n = Dropout(dropout)(n)
    n = Dense(units=neurons, activation='relu', name='ReLu')(n)
    n = Dense(units=1, activation='relu', name='output')(n)
    m = Model(i, n)
    m.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr), metrics=[RootMeanSquaredError(), 'mae'])
    return m


def train_model(id, model_folder, model:keras.Model, neurons, dropout, epochs, batch_size, x_train, y_train, lr):
    if not os.path.isdir(os.path.join(model_folder)):
        os.mkdir(os.path.join(model_folder))
    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=100),
        ModelCheckpoint(
            monitor='val_loss', save_best_only=True, mode='min',
            filepath=model_folder + '/{}/lstm_neur{}-do{}-ep{}-bs{}-lr{}.h5'.format(id, neurons, dropout, epochs, batch_size, lr))
    ]

    '''model = Sequential()
    model.add(LSTM(units=neurons, input_shape=x_train.shape))
    model.add(Dropout(dropout))
    model.add(Dense(units=1, activation='linear', name='ReLu'))
    #model.add(Dense(units=1, activation='relu', name='output'))'''

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.002, verbose=0,
                        shuffle=False, callbacks=callbacks)
    return model, history


if __name__ == "__main__":
    train_dir = "single_target_datasets/train"
    test_dir = "single_target_datasets/test"

    with open("single_target_results.txt", 'w') as f:
        f.write("      MAE    RMSE\n")
        f.close()
    for f in tqdm(sorted(os.listdir(train_dir))):
        if f == ".csv":
            continue
        id = f.split('.')[0]
        train = pd.read_csv(os.path.join(train_dir, f))
        test = pd.read_csv(os.path.join(test_dir, f))

        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train, y_train = create_lstm_tensors(train, scaler, '12')
        x_test, y_test = create_lstm_tensors(test, scaler, '12')

        neurons = 12
        dropout = 0.3
        lr = 0.005
        model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
        model, hist = train_model(id, model_folder="single_target_models", model=model, epochs=200, batch_size=12,
                                  x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr)

        predictions = model.predict(x_test)
        #predictions = scaler.inverse_transform(predictions)
        #inversed_y = scaler.inverse_transform(y_test)
        rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
        with open("single_target_results.txt", 'a') as f:
            f.write("%s: %s  %s\n"%(id, mae, rmse))
