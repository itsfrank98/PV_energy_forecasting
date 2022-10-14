""" File containing the functions to create the single target and the multitarget models """
import sys
sys.path.append('../')
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
from utils import sort_results


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
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.002, verbose=0,
                        shuffle=False, callbacks=callbacks)
    return model, history

def compute_results(predictions, y_test, file_name, id):
    """
    Function that calculates the evaluation metrics and writes the results on a file
    """
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
    mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
    with open(file_name, 'a') as f:
        f.write("%s: %s  %s\n"%(id, mae, rmse))

def test_model_multi_target(predictions, y_test, file_name, ids):
    """
    :param predictions: Predictions of the model
    :param y_test: Actual values
    :param file_name: Name on the file where the results will be written
    :param ids: IDs of the plants that the model considers.
    """
    j = 0
    for i in range(len(ids)):
        compute_results(predictions[j:j+12, :], y_test[j:j+12, :], file_name, ids[i])
        j += 12


def main(args):
    train_dir = args.train_dir
    test_dir = args.test_dir
    file_name = args.file_name
    neurons = args.neurons
    dropout = args.dropout
    lr = args.lr
    epochs = args.epochs
    model_type = args.model_type
    model_folder = args.model_folder

    os.makedirs(model_folder, exist_ok=True)

    for f in tqdm(sorted(os.listdir(train_dir))):
        if f == ".csv" or f.endswith(".txt"):
            continue
        ids = [f.split('.')[0]]
        train = pd.read_csv(os.path.join(train_dir, f))
        test = pd.read_csv(os.path.join(test_dir, f))
        create_lstm_tensors_minmax(train, None)
        x_train, y_train, scaler = create_lstm_tensors_minmax(train, None)
        x_test, y_test, _ = create_lstm_tensors_minmax(test, scaler)
        x_test[x_test<0] = 0
        x_test[x_test>1] = 1
        y_test[y_test<0] = 0
        y_test[y_test>1] = 1
        if model_type == "single_target":
            model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
        elif model_type == "multi_target":
            '''with open(train_dir+"/ids.txt") as f:
                for line in f:
                    ids = line.split("_")'''
            ids = f.split(".")[0].split('_')
            model = create_multi_target_model(neurons=neurons, dropout=dropout, x_train=x_train, ids=ids, lr=lr)
        else:
            raise ValueError("Invalid model type, it can only be 'single_target' or 'multi_target'")

        model, hist = train_model(ids, model_folder=model_folder, model=model, epochs=epochs, batch_size=12,
                                  x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr)
        predictions = model.predict(x_test)
        if model_type == "multi_target" and len(ids)>1:    # If there is only one element in the cluster it's like a single target model
            test_model_multi_target(np.vstack(np.array(predictions)), y_test, file_name, ids)
        else:
            compute_results(predictions, y_test, file_name, ids[0])

    #sort_results("r.txt", file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training directory")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the testing directory")
    parser.add_argument("--file_name", type=str, required=True, help="Name of the file where the results will be written")
    parser.add_argument("--neurons", type=int, required=True, help="Number of neurons")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--model_folder", type=str, required=True, help="Folder where the models will be saved")
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model to create. It can either be 'single_target' or 'multi_target'",
                        choices=['single_target', 'multi_target'])

    args = parser.parse_args()
    main(args)
    '''train_dir = "multitarget_1_space/train"
    test_dir = "multitarget_1_space/test"
    model = keras.models.load_model("multitarget_1_space/lstm_neur12-do0.3-ep200-bs12-lr0.005.h5")
    for f in tqdm(sorted(os.listdir(train_dir))):
        if f == ".csv" or f.endswith(".txt"):
            continue
        train = pd.read_csv(os.path.join(train_dir, f))
        test = pd.read_csv(os.path.join(test_dir, f))
        x_train, y_train, scaler = create_lstm_tensors_minmax(train, None)
        x_test, y_test, _ = create_lstm_tensors_minmax(test, scaler)
        x_test[x_test<0] = 0
        x_test[x_test>1] = 1
        y_test[y_test<0] = 0
        y_test[y_test>1] = 1
        with open(train_dir+"/ids.txt") as f:
            for line in f:
                ids = line.split("_")
        predictions = model.predict(x_test)
        print(np.array(predictions).shape)
        #test_model_multi_target(predictions, y_test, "multitarget_1_space/results1.txt", ids)
'''
# python models.py --train_dir multitarget_15_space/train --test_dir multitarget_15_space/test --file_name multitarget_15_space/results1.txt --neurons 12 --dropout 0.3 --lr 0.005 --model_folder multitarget_15_space/models --model_type multi_target --epochs 200
