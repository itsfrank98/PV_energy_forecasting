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
from preprocess import create_lstm_tensors_minmax
import pandas as pd
import tensorflow as tf
import os
import numpy as np
import random as rn
from tqdm import tqdm
import argparse
from utils import sort_results, load_from_pickle


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

def train_separate_models(train_dir, test_dir, model_type, neurons, dropout, model_folder, epochs, lr, aggregate_training, clustering_dictionary:dict):
    """
    Function that trains one model for each plant or a multitarget model
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
        x_train, y_train, scaler = create_lstm_tensors_minmax(train, None, aggregate_training=aggregate_training)
        x_test, y_test, _ = create_lstm_tensors_minmax(test, scaler, aggregate_training=aggregate_training)
        if model_type == "single_target":
            model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
        elif model_type == "multi_target":
            model = create_multi_target_model(neurons=neurons, dropout=dropout, x_train=x_train, ids=ids, lr=lr)

        model, hist = train_model(f, model_folder=model_folder, model=model, epochs=epochs, batch_size=12,
                                  x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr)
        predictions = model.predict(x_test)
        '''x_test = pd.DataFrame(x_test.reshape((12, 12)))
        predictions = x_test.mean(axis=1)'''
        if model_type == "multi_target":
            test_model_multi(np.vstack(np.array(predictions)), y_test, "r.txt", ids)
        else:
            compute_results(predictions, y_test, "r.txt", ids)

def train_unique_model(train_dir, test_dir, neurons, dropout, model_folder, epochs, lr):
    """
    Function that trains a unique model using data coming for all the plants
    """
    train = pd.DataFrame()
    for f in sorted(os.listdir(train_dir)):
        if f == ".csv" or f.endswith(".txt"):
            continue
        train = pd.concat((train, pd.read_csv(os.path.join(train_dir, f))))
    x_train, y_train, scaler = create_lstm_tensors_minmax(train, scaler=None, aggregate_training=False)

    model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
    #model = keras.models.load_model("single_target/unique/lstm_neur12-do0.3-ep200-bs12-lr0.003.h5")
    model, hist = train_model("unique", model_folder=model_folder, model=model, epochs=epochs, batch_size=12,
                              x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr)
    avg = np.mean(y_train)  # Average of the target labels in the training set. It will be used to compute the relative squared error

    for f in tqdm(sorted(os.listdir(test_dir))):
        id = f.split('.')[0]
        if f == ".csv" or f.endswith(".txt"):
            continue
        test = pd.read_csv(os.path.join(test_dir, f))
        x_test, y_test, _ = create_lstm_tensors_minmax(test, scaler=scaler, aggregate_training=False)
        pred = model.predict(x_test)
        test_model_multi(np.vstack(np.array(pred)), y_test, "r.txt", [id])  # Calculate MAE and RMSE

        # The following is done for computing RSE. Predictions for each plant are stacked in a unique array, as well as the actual values
        if id == "0":
            predictions = pred
            t = y_test
        else:
            predictions = np.vstack((predictions, pred))
            t = np.vstack((t, y_test))

    predictions_avg = np.zeros(predictions.shape[0])+avg
    rse = compute_rse(predictions, predictions_avg, t, "r.txt", id)
    print(rse)


def train_single_model_clustering(train_dir, test_dir, neurons, dropout, model_folder, epochs, lr):
    """Train a separate single target model for each cluster of points"""
    for f in tqdm(sorted(os.listdir(train_dir))):
        if f == ".csv" or f.endswith(".txt"):
            continue
        ids = f.split('.')[0]
        train = pd.read_csv(os.path.join(train_dir, f))
        test = pd.read_csv(os.path.join(test_dir, f))
        x_train, y_train, scaler = create_lstm_tensors_minmax(train, scaler=None, aggregate_training=False)
        x_test, y_test, _ = create_lstm_tensors_minmax(test, scaler=scaler, aggregate_training=None)
        model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
        model, hist = train_model(ids, model_folder=model_folder, model=model, epochs=epochs, batch_size=12,
                                  x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr)
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

def main(args):
    f = open("r.txt", 'r+')     # r.txt is a utility file where the results will be reported. Then they will be sorted alphabetically according to the plant IDs and written on the file that the user indicated
    # Delete the previous content from r.txt
    f.seek(0)
    f.truncate()
    f.close()

    train_dir = args.train_dir
    test_dir = args.test_dir
    file_name = args.file_name
    neurons = args.neurons
    dropout = args.dropout
    lr = args.lr
    epochs = args.epochs
    model_folder = args.model_folder
    training_type = args.training_type
    aggregate_training = args.aggregate_training
    clustering_dictionary = args.clustering_dictionary

    os.makedirs(model_folder, exist_ok=True)
    if training_type == "single_model_clustering":
        train_single_model_clustering(train_dir, test_dir, neurons, dropout, model_folder, epochs, lr)
    elif training_type == "single_model":
        train_unique_model(train_dir, test_dir, neurons, dropout, model_folder, epochs, lr)
    elif training_type == "multi_target":
        clustering_dict = load_from_pickle(clustering_dictionary)
        train_separate_models(train_dir=train_dir, test_dir=test_dir, model_type=training_type, neurons=neurons, dropout=dropout,
                              model_folder=model_folder, epochs=epochs, lr=lr, aggregate_training=aggregate_training, clustering_dictionary=clustering_dict)
    elif training_type == "single_target":
        train_separate_models(train_dir=train_dir, test_dir=test_dir, model_type=training_type, neurons=neurons, dropout=dropout,
                              model_folder=model_folder, epochs=epochs, lr=lr, aggregate_training=aggregate_training, clustering_dictionary=None)
    sort_results("r.txt", file_name)


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
    parser.add_argument("--training_type", type=str, required=True, help="Type of the model to create. It can either be:\n"
                                                                         " -'single_target' to train one model for each plant\n"
                                                                         " -'multi_target' to train a multitarget model for each plant cluster\n"
                                                                         " -'single_model' to train a unique model with data coming from all the plants\n"
                                                                         " -'single_model_clustering' to train, for each cluster of plants, a unique model with data coming from all the plants in that cluster",
                        choices=['single_target', 'multi_target', 'single_model', 'single_model_clustering'])
    parser.add_argument("--aggregate_training", required=False, type=bool, help="Set this to true if you are training on the aggregated dataset")
    parser.add_argument("--clustering_dictionary", required="--argument" in sys.argv or "single_model_clustering" in sys.argv or "multi_target" in sys.argv, type=str,
                        help="Path to the clustering dictionary. Needed only if training_type==single_model_clustering or training_type==multi_target")

    args = parser.parse_args()
    main(args)
    '''m = keras.models.load_model("single_target/models/unscaled_y/0.0.csv/lstm_neur12-do0.3-ep200-bs12-lr0.005.h5")
    train = pd.read_csv("single_target/train/0.0.csv")
    test = pd.read_csv("single_target/test/0.0.csv")
    _, _, scaler = create_lstm_tensors_minmax(train, None, aggregate_training=None)
    x, y, _ = create_lstm_tensors_minmax(test, scaler, aggregate_training=None)
    predictions = m.predict(x)
    print(predictions)
    print(y)'''

# python models.py --train_dir multitarget_15_space_BEST/train1 --test_dir multitarget_15_space_BEST/test --file_name multitarget_15_space_BEST/results1.txt --neurons 12 --dropout 0.3 --lr 0.005 --model_folder multitarget_15_space_BEST/models --training_type multi_target --epochs 200

