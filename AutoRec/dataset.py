import os
import zipfile
import requests
import hparams
import numpy as np
import pandas as pd
import torch.utils.data as data


class AutoRecDataset(data.Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.data_df: np.ndarray = self.get_dataset(self.is_train)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df[index, :]

    def download_dataset(self):
        if os.path.isdir(hparams.save_path):
            print("Data already exists. Start training directly.")
            return
        else:
            os.mkdir(hparams.save_path)
            print("Data file does not exists. Start downloading..")

        get_response = requests.get(hparams.data_url)
        file_name = hparams.data_url.split("/")[-1]
        with open(os.path.join("./data", file_name), "wb") as f:
            for chunk in get_response.iter_content(chunk_size=None):
                f.write(chunk)
        print("Movielens dataset (ml-latest-small.zip) download completed.")

        with zipfile.ZipFile(hparams.data_path, mode="r") as z:
            z.extractall(hparams.save_path)
        print("ml-latest-small.zip extraction(unzip) done.")

        os.remove(hparams.data_path)
        print("Removing zip file completed.")

    def get_dataset(self, is_train):

        self.download_dataset()

        df: pd.DataFrame = pd.read_csv(
            os.path.join(hparams.data_base_path, "ratings.csv"),
        )
        df.drop(columns=["timestamp"], inplace=True)

        user2idx, item2idx = {}, {}
        for idx, user in enumerate(df["userId"].unique().tolist()):
            user2idx[user] = idx

        for idx, item in enumerate(df["movieId"].unique().tolist()):
            item2idx[item] = idx

        df["userId"] = df["userId"].map(user2idx)
        df["movieId"] = df["movieId"].map(item2idx)

        n_transaction = len(df["userId"])
        train_idx = int(n_transaction * 0.8)
        n_user = df["userId"].nunique()
        n_item = df["movieId"].nunique()

        if is_train:
            df = df.iloc[:train_idx, :]
        else:
            df = df.iloc[train_idx:, :]

        data_mat = np.zeros((n_user, n_item))
        for _, (user, item, rating) in df.iterrows():
            user = int(user)
            item = int(item)
            data_mat[user, item] = rating

        return data_mat
