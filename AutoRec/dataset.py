import os
import zipfile
import requests
import hparams
import pandas as pd
import torch.utils.data as data


class AutoRecDataset(data.Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.data_df = self.get_dataset(self.is_train)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df[index]

    def download_dataset(self):
        if os.path.isdir(hparams.save_path):
            print("Data already exists. Start training directly.")
            return
        else:
            os.mkdir(hparams.save_path)
            print("Data file does not exists. Start downloading..")

        get_response = requests.get(hparams.data_url)
        file_name = hparams.data_url.split("/")[-1]
        with open(os.path.join('./data', file_name), 'wb') as f:
            for chunk in get_response.iter_content(chunk_size=None):
                f.write(chunk)
        print("Movielens dataset (ml-latest-small.zip) download completed.")

        with zipfile.ZipFile(hparams.data_path, mode='r') as z:
            z.extractall(hparams.save_path)
        print("ml-latest-small.zip extraction(unzip) done.")

        os.remove(hparams.data_path)
        print("Removing zip file completed.")

    def get_dataset(self, is_train):
        df: pd.DataFrame = pd.read_csv(os.path.join(hparams.data_base_path, 'ratings.csv')).to_numpy()[:, :3]
        user_list = df['userId'].unique().tolist()
        n_user = len(user_list)

        train_user = user_list[:n_user * 0.8]
        test_user = user_list[n_user * 0.8:]

        if is_train:
            df = df[df['userId'] in train_user]
        else:
            df = df[df['userId'] in test_user]

        return df