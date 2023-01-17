import torch

class CFG:
    gpu_index = 0
    device    = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
    top_k     = 10
    seed      = 42
    neg_ratio = 0.1
    test_size = 0.2
    data_path = "./data/"
    data_name = "ml-latest-small"
    data_url  = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
