import os
import hparams
import zipfile
import requests
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


def download_dataset():
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


def to_implicit_feedback(target):
    target[target <= 3] = 0
    target[target > 3] = 1
    return target


class FMDataset(Dataset):
    def __init__(self):
        super(FMDataset, self).__init__()
        download_dataset()
        self.data = pd.read_csv(os.path.join(hparams.data_base_path, 'ratings.csv')).to_numpy()[:, :3]

        self.input = self.data[:, :2]
        self.target = to_implicit_feedback(self.data[:, 2])
        self.field_dims = np.max(self.input, axis=0) + 1

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index]
