"""Model having as dataset the one where there are the features and the means of the features in the same cluster"""
from preprocess import create_lstm_tensors
from models import create_single_target_model, train_model, compute_results
import os
import pandas as pd
import argparse
import sys
sys.path.append('../')
from utils import sort_results, load_from_pickle


def train_and_test_models(train_dir, test_dir, neurons, dropout, model_folder, epochs, lr, preprocess, y_column, batch_size, patience):
    train = pd.DataFrame()
    for f in sorted(os.listdir(train_dir)):
        if f == ".csv" or f.endswith(".txt"):
            continue
        train = pd.concat((train, pd.read_csv(os.path.join(train_dir, f))))
    x_train, y_train, scaler = create_lstm_tensors(train, scaler=None, y_column=y_column, preprocess=preprocess)

    #model = keras.models.load_model("../latiano/multitarget_vertical/time/20/models/unique_aggregate/lstm_neur12-do0.3-ep200-bs100-lr0.005.h5")
    model = create_single_target_model(neurons=neurons, dropout=dropout, x_train=x_train, lr=lr)
    model, hist = train_model("unique_aggregate", model_folder=model_folder, model=model, epochs=epochs, batch_size=batch_size,
                x_train=x_train, y_train=y_train, neurons=neurons, dropout=dropout, lr=lr, patience=patience)

    # Test the model on all the plants that are in the cluster covered by the model
    for f in sorted(os.listdir(test_dir)):
        test = pd.read_csv(os.path.join(test_dir, f))
        x_test, y_test, _ = create_lstm_tensors(df=test, scaler=scaler, preprocess=preprocess, y_column=y_column)
        evaluate(model, x_test, y_test, "r.txt", f)


def evaluate(model, x_test, y_test, file_name, id):
    predictions = model.predict(x_test)
    compute_results(predictions=predictions, y_test=y_test, file_name=file_name, id=id)


def main(args):
    f = open("model/r.txt", 'r+')
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
    batch_size = args.batch_size
    patience = args.patience
    clustering_dict_path = args.clustering_dict_path
    preprocess = args.preprocess
    y_column = args.y_column
    clustering_dict = load_from_pickle(clustering_dict_path)
    if preprocess == 0:
        prep = False
    else:
        prep = True
    train_and_test_models(train_dir=train_dir, test_dir=test_dir, neurons=neurons, dropout=dropout, model_folder=model_folder, epochs=epochs, lr=lr,
                          batch_size=batch_size, patience=patience, preprocess=prep, y_column=y_column)
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
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--patience", type=int, required=True, help="Patience")
    parser.add_argument("--model_folder", type=str, required=True, help="Folder where the models will be saved")
    parser.add_argument("--clustering_dict_path", required=True, type=str, help="Path to the clustering dictionary." )
    parser.add_argument("--preprocess", type=int, required=True, help="1 to perform feature scaling, 0 to not perform it", choices=[0, 1])
    parser.add_argument("--y_column", required=True, type=int, help="Name of the column having the target variable")
    args = parser.parse_args()
    main(args)
