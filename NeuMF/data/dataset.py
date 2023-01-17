import os

import pandas as pd

from cfg import CFG
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm

def download_dataset():
    if not Path(os.path.join(CFG.data_path, CFG.data_name)).exists():
        http_response = urlopen(CFG.data_url)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=CFG.data_path)


def prepare_dataset():
    rating_df = pd.read_csv(os.path.join(CFG.data_path, CFG.data_name, "ratings.csv"))
    rating_matrix = rating_df.pivot(index="userId", columns="movieId", values="rating")

    implicit_df = {}
    implicit_df["userId"] = []
    implicit_df["movieId"] = []
    implicit_df["implicit_feedback"] = []

    user_dict = {}
    movie_dict = {}

    user_ids = rating_df["userId"].unique()
    movie_ids = rating_df["movieId"].unique()

    for user, user_id in tqdm(enumerate(user_ids), leave=True):
        user_dict[user] = user_id
        for movie, movie_id in enumerate(movie_ids):
            if movie not in movie_dict:
                movie_dict[movie] = movie_id
            implicit_df["userId"].append(user)
            implicit_df["movieId"].append(movie)

            if pd.isna(rating_matrix.loc[user_id, movie_id]):
                implicit_df["implicit_feedback"].append(0)
            else:
                implicit_df["implicit_feedback"].append(1)
    
    implicit_df = pd.DataFrame(implicit_df)
    implicit_df["userId"].astype("category")
    implicit_df["movie_id"].astype("category")


