import sys
sys.path.append('../')
import keras.models
from preprocess import create_lstm_tensors_minmax
from models import create_single_target_model, train_model, compute_results
from tqdm import tqdm
import os
import pandas as pd
from utils import sort_results
import argparse
from utils import load_from_pickle

def train_models(train_dir, neurons, dropout, model_folder, epochs, lr):
    scalers_dict = {}
    for f in tqdm(sorted(os.listdir(train_dir))):
        if f == ".csv" or f.endswith(".txt"):
            continue
        fname = f.split('.')[0]
        train = pd.read_csv(os.path.join(train_dir, f))
        x_train, y_train, scaler = create_lstm_tensors_minmax(train, None, aggregate_training=True)
        scalers_dict[fname] = scaler
        model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
        train_model(fname, model_folder=model_folder, model=model, epochs=epochs, batch_size=12,
                    x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr)
    return scalers_dict     # We don't return the trained models since they're saved in the directory

def evaluate(model, x_test, y_test, file_name, id):
    predictions = model.predict(x_test)
    compute_results(predictions=predictions, y_test=y_test, file_name=file_name, id=id)


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
    clustering_dict_path = args.clustering_dict_path
    clustering_dict = load_from_pickle(clustering_dict_path)

    scalers_dict = train_models(train_dir=train_dir, neurons=neurons, dropout=dropout, model_folder=model_folder, epochs=epochs, lr=lr)
    for k in scalers_dict.keys():
        scaler = scalers_dict[k]
        ids = clustering_dict[int(k)]
        for id in ids:
            if id == "133.0":
                continue
            test = pd.read_csv(os.path.join(test_dir, id+'.csv'))
            x_test, y_test, _ = create_lstm_tensors_minmax(test, scaler, aggregate_training=True)
            model = keras.models.load_model(os.path.join(model_folder, k, "lstm_neur{}-do{}-ep{}-bs12-lr{}.h5".format(neurons, dropout, epochs, lr)))
            evaluate(model, x_test, y_test, "r.txt", id.split('.')[0])
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
    parser.add_argument("--clustering_dict_path", required=True, type=str, help="Path to the clustering dictionary." )

    args = parser.parse_args()
    main(args)
