import sys
import torch
import hparams
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dataset import FMDataset
from utils import plot_loss_graph
from model import FactorizationMachine
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


# Prepare dataset
dataset = FMDataset()
train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, random_state=42, shuffle=True)
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
        for X, y in tqdm_test:
            X = X.to(device)

            output = model(X).cpu()
            y = y.to(torch.float32)
            loss = criterion(output, y)

            epoch_test_loss += loss.item()
            tqdm_test.set_postfix(BCE_loss=loss.item())

    epoch_test_loss_history.append(epoch_test_loss / len(test_dataloader))
    print(f"Test Epoch #{epoch + 1} finished. Loss : {epoch_test_loss / len(test_dataloader): .4f}", flush=True)

# plot loss graph
plot_loss_graph(epoch_train_loss_history, epoch_test_loss_history)
