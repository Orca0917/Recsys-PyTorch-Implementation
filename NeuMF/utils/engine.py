import torch

from utils.util import save_checkpoint, use_optimizer
from utils.metric import MetronAtk


class Engine(object):
    def __init__(self, config):
        self.config = config
        self._metron = MetronAtk(top_k=10)
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        if self.config["use_cuda"] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print(f"[Training Epoch {epoch_id}] Batch {batch_id}, Loss {loss}")
            total_loss += loss

    def evaluate(self, evaluate_data, epoch_id):
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]

            if self.config["use_cuda"] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()

            test_scores = self.model(test_users, test_items)
            negative_scores = self.model(negative_users, negative_items)

            if self.config["use_cuda"] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()

            self._metron.subjects = [
                test_users.data.view(-1).tolist(),
                test_items.data.view(-1).tolist(),
                test_scores.data.view(-1).tolist(),
                negative_users.data.view(-1).tolist(),
                negative_items.data.view(-1).tolist(),
                negative_scores.data.view(-1).tolist(),
            ]

        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        print(f"[Evaluating Epoch {epoch_id}] HitRatio = {hit_ratio :.4f}, NDCG = {ndcg :.4f}")
        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        model_dir = self.config["model_dir"].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
