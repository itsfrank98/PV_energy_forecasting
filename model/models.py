""" File containing the functions to create the models """
import sys
sys.path.append('../')
import keras
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense, Input
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocess import create_lstm_tensors
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

def train_model(id, model_folder, model:keras.Model, neurons, dropout, epochs, batch_size, x_train, y_train, lr, patience):
    if not os.path.isdir(os.path.join(model_folder)):
        os.mkdir(os.path.join(model_folder))
    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=patience),
        ModelCheckpoint(
            monitor='val_loss', save_best_only=True, mode='min',
            filepath=model_folder + '/{}/lstm_neur{}-do{}-ep{}-bs{}-lr{}.h5'.format(id, neurons, dropout, epochs, batch_size, lr))
    ]
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.002, verbose=1,
                        shuffle=False, callbacks=callbacks)
    return model, history

def train_separate_models(train_dir, test_dir, model_type, neurons, dropout, model_folder, epochs, lr, y_column, preprocess,
                          patience, batch_size, clustering_dictionary:dict=None, step=0):
    """
    Function that trains one model for each plant or a multitarget model
    :param train_dir: Directory containing training files
    :param test_dir: Directory containing testing files
    :param model_type: Type of model to create
    :param neurons:
    :param dropout:
    :param model_folder: Folder where the models will be saved
    :param epochs:
    :param lr:
    :param aggregate_training: Set to true if we are performing aggregate training
    :param y_column:
    :param preprocess:
    :param clustering_dictionary:
    :param step:
    :return:
    """
    for f in tqdm(sorted(os.listdir(train_dir))):
        if f == ".csv" or f.endswith(".txt"):
            continue
        fname = f.split('.')[0]
        if clustering_dictionary:
            ids = clustering_dictionary[int(fname)]
        else:
            ids = fname
        train = pd.read_csv(os.path.join(train_dir, f))
        test = pd.read_csv(os.path.join(test_dir, f))
        x_train, y_train, scaler = create_lstm_tensors(train, None, preprocess=preprocess, y_column=y_column, step=step)
        x_test, y_test, _ = create_lstm_tensors(test, scaler, preprocess=preprocess, y_column=y_column, step=step)
        if model_type == "single_target":
            model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
            #model = keras.models.load_model(os.path.join(model_folder, f, "lstm_neur18-do0.3-ep200-bs200-lr0.005.h5"))
        elif model_type == "multi_target":
            model = create_multi_target_model(neurons=neurons, dropout=dropout, x_train=x_train, ids=ids, lr=lr)

        model, hist = train_model(f, model_folder=model_folder, model=model, epochs=epochs, batch_size=batch_size,
                                  x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr, patience=patience)
        predictions = model.predict(x_test)
        if model_type == "multi_target":
            test_model_multi(np.vstack(np.array(predictions)), y_test, "r.txt", ids)
        else:
            compute_results(predictions, y_test, "r.txt", ids)

def train_unique_model(train_dir, test_dir, neurons, dropout, model_folder, epochs, lr, y_column, preprocess, patience, batch_size):
    """
    Function that trains a unique model using data coming for all the plants
    """
    train = pd.DataFrame()
    for f in sorted(os.listdir(train_dir)):
        if f == ".csv" or f.endswith(".txt"):
            continue
        train = pd.concat((train, pd.read_csv(os.path.join(train_dir, f))))
    x_train, y_train, scaler = create_lstm_tensors(train, scaler=None, y_column=y_column, preprocess=preprocess)

    model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
    #model = keras.models.load_model("pvitaly/single_model/unique/lstm_neur18-do0.3-ep200-bs200-lr0.005.h5")
    model, hist = train_model("unique", model_folder=model_folder, model=model, epochs=epochs, batch_size=batch_size,
                              x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr, patience=patience)
    #avg = np.mean(y_train)  # Average of the target labels in the training set. It will be used to compute the relative squared error

    for f in tqdm(sorted(os.listdir(test_dir))):
        id = f.split('.')[0]
        if f == ".csv" or f.endswith(".txt"):
            continue
        test = pd.read_csv(os.path.join(test_dir, f))
        x_test, y_test, _ = create_lstm_tensors(test, scaler=scaler, y_column=y_column, preprocess=preprocess)
        pred = model.predict(x_test)
        test_model_multi(np.vstack(np.array(pred)), y_test, "r.txt", [id])  # Calculate MAE and RMSE

    '''    # The following is done for computing RSE. Predictions for each plant are stacked in a unique array, as well as the actual values
        if id == "0":
            predictions = pred
            t = y_test
        else:
            predictions = np.vstack((predictions, pred))
            t = np.vstack((t, y_test))

    predictions_avg = np.zeros(predictions.shape[0])+avg
    rse = compute_rse(predictions, predictions_avg, t, "r.txt", id)
    print(rse)'''


def train_single_model_clustering(train_dir, test_dir, neurons, dropout, model_folder, epochs, lr, y_column, preprocess, patience, batch_size):
    """Train a separate single target model for each cluster of points"""
    for f in tqdm(sorted(os.listdir(train_dir))):
        if f == ".csv" or f.endswith(".txt"):
            continue
        ids = f.split('.')[0]
        train = pd.read_csv(os.path.join(train_dir, f))
        test = pd.read_csv(os.path.join(test_dir, f))
        x_train, y_train, scaler = create_lstm_tensors(train, scaler=None, y_column=y_column, preprocess=preprocess)
        x_test, y_test, _ = create_lstm_tensors(test, scaler=scaler, y_column=y_column, preprocess=preprocess)
        model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
        model, hist = train_model(ids, model_folder=model_folder, model=model, epochs=epochs, batch_size=batch_size,
                                  x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr, patience=patience)
        predictions = model.predict(x_test)
        ids_list = ids.split('_')
        test_model_multi(np.vstack(np.array(predictions)), y_test, "r.txt", ids_list)


def compute_results(predictions, y_test, file_name, id):
    """
    Function that calculates the evaluation metrics and writes the results on a file
    """
    rmse = np.sqrt(mean_squared_error(y_true=y_test, y_pred=predictions))
    mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
    with open(file_name, 'a') as f:
        f.write("%s: %s  %s\n"%(id, mae, rmse))

def test_model_multi(predictions, y_test, file_name, ids):
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

def compute_rse(pred, pred_avg, actual, file_name, id):
    """
    Computes the residual squared error
    """
    sem = np.sum(pred - actual)**2
    sea = np.sum(actual - pred_avg)**2
    rse = sem / sea
    '''with open(file_name, 'a') as f:
        f.write("%s: %s \n"%(rse))'''
    return rse
