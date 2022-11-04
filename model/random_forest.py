import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import argparse
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models import compute_results
from tqdm import tqdm
from utils import sort_results, save_to_pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_lstm_tensors_minmax(df, scaler):
    data = df.values
    columns = np.arange(start=1, stop=len(df.columns))
    y_columns = np.arange(start=12, stop=len(df.columns)+1, step=13) + 1    # Every 12 columns we have the value of the target variable
    x_columns = [c for c in columns if c not in y_columns]
    if not scaler:
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(data)
    else:
        scaled = scaler.transform(data)
    x, y = scaled[:, x_columns], scaled[:, y_columns]
    x = x.reshape(x.shape[0], x.shape[1], 1)

    # If we are testing we need to deal with the out of range values
    if scaler:
        x[x<0] = 0
        x[x>1] = 1
        y[y<0] = 0
        y[y>1] = 1
    return x, y, scaler

def compute_results_multi(predictions, y_test, file_name, id):
    """
    Function that calculates the evaluation metrics and writes the results on a file
    """
    ids = id.split("_")
    for i in range(len(ids)):
        rmse = np.sqrt(mean_squared_error(y_true=y_test[:, i], y_pred=predictions[:, i]))
        mae = mean_absolute_error(y_true=y_test[:, i], y_pred=predictions[:, i])
        with open(file_name, 'a') as f:
            f.write("%s: %s  %s\n"%(ids[i], mae, rmse))

def train_single_target(train_dir, test_dir, model_folder, multi):
    for f in tqdm(os.listdir(train_dir)):
        if f == ".csv":
            continue
        #if f == "4_3.csv":
        train = pd.read_csv(os.path.join(train_dir, f))
        test = pd.read_csv(os.path.join(test_dir, f))
        train_x, train_y, scaler = create_lstm_tensors_minmax(train, None)
        test_x, test_y, _ = create_lstm_tensors_minmax(test, scaler)
        if multi:
            mod = MultiOutputRegressor(
                RandomForestRegressor(criterion='absolute_error', max_features='sqrt', random_state=0)
            )
        else:
            mod = RandomForestRegressor(criterion='absolute_error', max_features='sqrt', random_state=0)
        mod.fit(train_x.reshape(train_x.shape[0], train_x.shape[1]), train_y)
        pred = mod.predict(test_x.reshape(test_x.shape[0], test_x.shape[1]))
        save_to_pickle(model_folder+"/"+f.split('.')[0], mod)
        if multi:
            compute_results_multi(pred, test_y, 'r.txt', f.split('.')[0])
        else:
            compute_results(pred, test_y.reshape(1, len(test_y))[0], 'r.txt', f.split('.')[0])



def main(args):
    f = open("r.txt", 'r+')     # r.txt is a utility file where the results will be reported. Then they will be sorted alphabetically according to the plant IDs and written on the file that the user indicated
    # Delete the previous content from r.txt
    f.seek(0)
    f.truncate()
    f.close()


    train_dir = args.train_dir
    test_dir = args.test_dir
    model_folder = args.model_folder
    type = args.training_type
    os.makedirs(model_folder, exist_ok=True)

    multi = False
    if type == "multi_target":
        multi = True
    train_single_target(train_dir, test_dir, model_folder, multi)
    sort_results("r.txt", args.file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training directory")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the testing directory")
    parser.add_argument("--file_name", type=str, required=True, help="Name of the file where the results will be written")
    parser.add_argument("--model_folder", type=str, required=True, help="Folder where the models will be saved")
    parser.add_argument("--training_type", type=str, required=True, help="Type of the model to create. It can either be:\n"
                                                                         " -'single_target' to train one model for each plant\n"
                                                                         " -'multi_target' to train a multitarget model for each plant cluster\n",
                        choices=['single_target', 'multi_target', 'single_model_clustering'])
    args = parser.parse_args()
    main(args)


