import sys
import torch
import hparams
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dataset import FMDataset
from utils import plot_loss_graph
from model import FactorizationMachine
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


# Prepare dataset
dataset = FMDataset()
train_data_len = int(len(dataset) * 0.8)
test_data_len = len(dataset) - train_data_len
train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_data_len, test_data_len))
train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)

# Model & Train settings.
model = FactorizationMachine(field_dims=dataset.field_dims, embedding_dim=hparams.embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=1e-5)
criterion = nn.BCELoss()

# GPU Settings.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# History recorder
epoch_train_loss_history = []
epoch_test_loss_history = []

# Training
for epoch in range(hparams.num_epochs):
    epoch_train_loss = 0.0

    model.train()
    tqdm_train = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Train Epoch #{epoch + 1}", file=sys.stdout)
    for X, y in tqdm_train:
        X = X.to(device)

        output = model(X).cpu()
        y = y.to(torch.float32)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        tqdm_train.set_postfix(BCE_loss=loss.item())

    epoch_train_loss_history.append(epoch_train_loss / len(train_dataloader))
    print(f"Train Epoch #{epoch + 1} finished. Loss : {epoch_train_loss / len(train_dataloader): .4f}", flush=True)

    # valid / test
    epoch_test_loss = 0.0
    model.eval()
    tqdm_test = tqdm(test_dataloader, total=len(test_dataloader), desc=f"Test Epoch #{epoch + 1}", file=sys.stdout)
    with torch.no_grad():
        targets, predicts = [], []
        for X, y in tqdm_test:
            X = X.to(device)

            output = model(X).cpu()
            y = y.to(torch.float32)

            predicts.extend(output.tolist())
            targets.extend(y.tolist())

        score = roc_auc_score(targets, predicts)
        tqdm_test.set_postfix(AUCROC=score)

    epoch_test_loss_history.append(score)
    print(f"Test Epoch #{epoch + 1} finished. AUCROC : {score: .4f}", flush=True)

# plot loss graph
plot_loss_graph(epoch_train_loss_history, epoch_test_loss_history)
