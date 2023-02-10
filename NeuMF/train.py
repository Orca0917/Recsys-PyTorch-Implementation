import pandas as pd
import numpy as np

from model.GMF import GMFEngine
from model.MLP import MLPEngine
from model.NeuMF import NeuMFEngine
from data.dataset import SampleGenerator


gmf_config = {
    "alias": "gmf_factor8_neg4-implicit",
    "num_epoch": 200,
    "batch_size": 1024,
    # 'optimizer': 'sgd',
    # 'sgd_lr': 1e-3,
    # 'sgd_momentum': 0.9,
    # 'optimizer': 'rmsprop',
    # 'rmsprop_lr': 1e-3,
    # 'rmsprop_alpha': 0.99,
    # 'rmsprop_momentum': 0,
    "optimizer": "adam",
    "adam_lr": 1e-3,
    "num_users": 610,
    "num_itmes": 9724,
    "latent_dim": 8,
    "num_negative": 4,
    "l2_regularization": 0,
    "use_cuda": True,
    "device_id": 0,
    "model_dir": "checkpoints/{}_HR{:.4f}_NDCG{:.4f}.model",
}

mlp_config = {
    "alias": "mlp_factor9_neg4-implicit",
    "num_epoch": 200,
    "batch_size": 256,  # 1024,
    "optimizer": "adam",
    "adam_lr": 1e-3,
    "num_users": 610,
    "num_items": 9724,
    "latent_dim": 8,
    "num_negative": 4,
    # layers[0] is the concat of latent user vector & latent item vector
    "layers": [16, 64, 32, 16, 8],
    "l2_regularization": 0.0000001,  # MLP model is sensitive to hyper params
    "use_cuda": True,
    "device_id": 7,
    "pretrain": True,
    "pretrain_mf": "checkpoints/{}".format("gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model"),
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

neumf_config = {
    "alias": "pretrain_neumf_factor8neg4",
    "num_epoch": 200,
    "batch_size": 1024,
    "optimizer": "adam",
    "adam_lr": 1e-3,
    "num_users": 610,
    "num_items": 9724,
    "latent_dim_mf": 8,
    "latent_dim_mlp": 8,
    "num_negative": 4,
    # layers[0] is the concat of latent user vector & latent item vector
    "layers": [16, 32, 16, 8],
    "l2_regularization": 0.01,
    "use_cuda": True,
    "device_id": 7,
    "pretrain": True,
    "pretrain_mf": "checkpoints/{}".format("gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model"),
    "pretrain_mlp": "checkpoints/{}".format("mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model"),
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

# 데이터 로드
ml1m_dir = "data/ml-latest-small/ratings.csv"
ml1m_rating = pd.read_csv(
    ml1m_dir, sep=",", header=None, names=["uid", "mid", "rating", "timestamp"], engine="python"
)

# 데이터 인덱스 재설정
user_id = ml1m_rating[["uid"]].drop_duplicates().reindex()
user_id["userId"] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=["uid"], how="left")

item_id = ml1m_rating[["mid"]].drop_duplicates()
item_id["itemId"] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=["mid"], how="left")

ml1m_rating = ml1m_rating[["userId", "itemId", "rating", "timestamp"]]
print(f"Range of userId is [{ml1m_rating.userId.min()}, {ml1m_rating.userId.max()}]")
print(f"Range of itemId is [{ml1m_rating.itemId.min()}, {ml1m_rating.itemId.max()}]")

# 데이터 로더
sample_generator = SampleGenerator(ratings=ml1m_rating)
evaluate_data = sample_generator.evaluate_data

# 모델 설정
config = neumf_config
engine = NeuMFEngine(config)

# 학습
for epoch in range(config["num_epoch"]):
    print(f"Epoch {epoch} starts")
    print("-" * 80)
    train_loader = sample_generator.instance_a_train_loader(
        config["num_negative"], config["batch_size"]
    )
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch=epoch)
    engine.save(config["alias"], epoch, hit_ratio, ndcg)
