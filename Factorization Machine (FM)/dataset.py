import os
import hparams
import zipfile
import requests
import pandas as pd

from torch.utils.data import Dataset


def download_dataset():
    if os.path.isdir(hparams.save_path):
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


class FMDataset(Dataset):
    def __init__(self, is_train: bool):
        super(FMDataset, self).__init__()
        download_dataset()
        self.is_train = is_train
        self.data = pd.read_csv(os.path.join(hparams.data_base_path, 'ratings.csv'))

        self.X = self.data.drop(['rating'])
        self.y = self.data['rating']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X.iloc[index], self.y.iloc[index]


