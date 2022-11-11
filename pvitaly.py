import os
import pandas as pd
from tqdm import tqdm
d = pd.read_csv("datasets/pvitaly/pv_italy_hourly.csv")
d = d[["idsito", "lat", "lon", "data", "kwh"]]
d = d.sort_values(by ='idsito')
ids = list(set(d['idsito']))
i = 0.0
columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
splitted_dst = "datasets/pvitaly/splitted"
windows_dst = "datasets/pvitaly/windows"
train_dir = "model/pvitaly/single_target/train"
test_dir = "model/pvitaly/single_target/test"
# Split the dataset into a separate csv file for every plant
for id in ids:
    df_id = d.loc[d['idsito']==id]
    df_id = df_id.sort_values("data")
    df_id = df_id.dropna()
    df_id.to_csv(os.path.join(splitted_dst, str(i)+".csv"))
    i += 1
for f in os.listdir(splitted_dst):
    df = pd.read_csv(os.path.join(splitted_dst, f))
    new_df = pd.DataFrame(columns=columns)
    for i in tqdm(range(len(df)-18)):
        dict = {str(j): df.iloc[i+j]['kwh'] for j in range(18)}  # Create a window of size 18 containing the productions for each hour
        new_df = new_df.append(dict, ignore_index=True)
    test_size = int(len(new_df)*0.2)
    train = new_df[:-test_size]
    test = new_df[-test_size:]
    train.to_csv(os.path.join(train_dir, f))
    test.to_csv(os.path.join(test_dir, f))

# python models.py --train_dir multitarget_15_space_BEST/train1 --test_dir multitarget_15_space_BEST/test --file_name multitarget_15_space_BEST/results1.txt --neurons 12 --dropout 0.3 --lr 0.005 --model_folder multitarget_15_space_BEST/models --training_type multi_target --epochs 200

