""" File containing the functions to create the single target and the multitarget models """
import keras
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense, Input
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prepare_data import create_lstm_tensors_minmax
import pandas as pd
import tensorflow as tf
import os
import numpy as np
import random as rn
from tqdm import tqdm
import argparse

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

def create_multi_target_model(neurons, dropout, x_train, ids, lr):
    input = Input(shape=(x_train.shape[1], x_train.shape[2]))
    output_layers = []
    losses = ['mean_squared_error'] * len(ids)
    loss_weights = [1.0] * len(ids)
    metrics_dict = {}
    for id in ids:
        n = LSTM(units=neurons)(input)
        n = Dropout(dropout)(n)
        n = Dense(units=neurons, activation='relu', name='ReLu_{}'.format(id))(n)
        n = Dense(units=1, activation='relu', name='output_{}'.format(id))(n)
        output_layers.append(n)
        metrics_dict['output_{}'.format(id)] = [RootMeanSquaredError(), 'mae']
    m = Model(inputs=input, outputs=output_layers)
    tf.keras.utils.plot_model(m, to_file="/tmp/mod.png", show_shapes=True)
    m.compile(loss=losses, optimizer=Adam(learning_rate=lr), loss_weights=loss_weights, metrics=metrics_dict)
    #print(m.summary())
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

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.002, verbose=1,
                        shuffle=False)
    return model, history

def main(args):
    train_dir = args.train_dir
    test_dir = args.test_dir
    file_name = args.file_name
    neurons = args.neurons
    dropout = args.dropout
    lr = args.lr
    model_type = args.model_type
    model_folder = args.model_folder

    os.makedirs(model_folder, exist_ok=True)

    with open(file_name, 'w') as f:
        f.write("      MAE    RMSE\n")
        f.close()
    for f in tqdm(sorted(os.listdir(train_dir))):
        if f == ".csv":
            continue
        id = f.split('.')[0]
        train = pd.read_csv(os.path.join(train_dir, f))
        test = pd.read_csv(os.path.join(test_dir, f))
        x_train, y_train, scaler = create_lstm_tensors_minmax(train, None)
        x_test, y_test, _ = create_lstm_tensors_minmax(test, scaler)
        x_test[x_test<0] = 0
        x_test[x_test>1] = 1
        y_test[y_test<0] = 0
        y_test[y_test>1] = 1
        if model_type == "single_target":
            model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
        elif model_type == "multi_target":
            ids = f.split(".")[0].split('_')
            model = create_multi_target_model(neurons=neurons, dropout=dropout, x_train=x_train, ids=ids, lr=lr)
        else:
            raise ValueError("Invalid model type, it can only be 'single_target' or 'multi_target'")
        model, hist = train_model(id, model_folder=model_folder, model=model, epochs=1, batch_size=12,
                                  x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr)

        predictions = model.predict(x_test)
        if model_type == "multi_target":
            i = 0
            j = 0
            pred = []
            while i < len(ids):
                pred.append(predictions[i][j:j+12])
                j += 12
                i += 1
        print(pred)
        #print(predictions)
        rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
        mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
        with open(file_name, 'a') as f:
            f.write("%s: %s  %s\n"%(id, mae, rmse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training directory")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the testing directory")
    parser.add_argument("--file_name", type=str, required=True, help="Name of the file where the results will be written")
    parser.add_argument("--neurons", type=int, required=True, help="Number of neurons")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--model_folder", type=str, required=True, help="Folder where the models will be saved")
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model to create. It can either be 'single_target' or 'multi_target'",
                        choices=['single_target', 'multi_target'])
    args = parser.parse_args()
    main(args)
