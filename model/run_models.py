import sys
sys.path.append('../')
import argparse
from utils import load_from_pickle, sort_results
from model.models import train_single_model_clustering, train_unique_model, train_separate_models
import os

def main(args):
    f = open("r.txt", 'r+')     # r.txt is a utility file where the results will be reported. Then they will be sorted alphabetically according to the plant IDs and written on the file that the user indicated
    # Delete the previous content from r.txt
    f.seek(0)
    f.truncate()
    f.close()

    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    file_name = args.file_name
    neurons = args.neurons
    dropout = args.dropout
    lr = args.lr
    epochs = args.epochs
    model_folder = args.model_folder
    training_type = args.training_type
    clustering_dictionary = args.clustering_dict_path
    patience = args.patience
    batch_size = args.batch_size
    y_column = args.y_column
    preprocess = args.preprocess

    if preprocess == 0:
        prep = False
    else:
        prep = True

    if training_type == "multi_target":
        if train_data_path.__contains__("pvitaly"):
            step = 18
        elif train_data_path.__contains__("latiano"):
            step = 13
    os.makedirs(model_folder, exist_ok=True)

    if training_type == "single_model_clustering":
        train_single_model_clustering(train_data_path, test_data_path, neurons, dropout, model_folder, epochs, lr, y_column=y_column,
                                      preprocess=prep, patience=patience, batch_size=batch_size)
    elif training_type == "single_model":
        train_unique_model(train_data_path, test_data_path, neurons, dropout, model_folder, epochs, lr, y_column=y_column,
                           preprocess=prep, patience=patience, batch_size=batch_size)
    elif training_type == "multi_target":
        clustering_dict = load_from_pickle(clustering_dictionary)
        train_separate_models(train_dir=train_data_path, test_dir=test_data_path, model_type=training_type, neurons=neurons, dropout=dropout,
                              model_folder=model_folder, epochs=epochs, lr=lr, y_column=y_column, preprocess=prep,
                              clustering_dictionary=clustering_dict, step=step, patience=patience, batch_size=batch_size)
    elif training_type == "single_target":
        train_separate_models(train_dir=train_data_path, test_dir=test_data_path, model_type=training_type, neurons=neurons, dropout=dropout,
                              model_folder=model_folder, epochs=epochs, lr=lr, preprocess=prep, y_column=y_column, patience=patience, batch_size=batch_size)
    sort_results("r.txt", file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training directory")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the testing directory")
    parser.add_argument("--file_name", type=str, required=True, help="Name of the file where the results will be written")
    parser.add_argument("--neurons", type=int, required=True, help="Number of neurons")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--patience", type=int, required=True, help="Patience")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--model_folder", type=str, required=True, help="Folder where the models will be saved")
    parser.add_argument("--training_type", type=str, required=True, help="Type of the model to create. It can either be:\n"
                                                                         " -'single_target' to train one model for each plant\n"
                                                                         " -'multi_target' to train a multitarget model for each plant cluster\n"
                                                                         " -'single_model' to train a unique model with data coming from all the plants\n"
                                                                         " -'single_model_clustering' to train, for each cluster of plants, a unique model with data coming from all the plants in that cluster",
                        choices=['single_target', 'multi_target', 'single_model', 'single_model_clustering'])
    parser.add_argument("--preprocess", type=int, required=True, help="1 to perform feature scaling, 0 to not perform it", choices=[0, 1])
    parser.add_argument("--y_column", required=True, type=int, help="Name of the column having the target variable")
    parser.add_argument("--clustering_dict_path", required="--argument" in sys.argv or "single_model_clustering" in sys.argv or "multi_target" in sys.argv, type=str,
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
