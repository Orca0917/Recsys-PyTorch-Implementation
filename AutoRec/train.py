import torch
import hparams
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from model import AutoRec
from dataset import AutoRecDataset
from torch.utils.data import DataLoader


# load dataset
train_dataset = AutoRecDataset()
test_dataset = AutoRecDataset(is_train=False)

# make dataloader
train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)

# model, optimizer definition
model = AutoRec(input_dim=hparams.input_dim, hidden_dim=hparams.hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
criterion = nn.MSELoss()

for epoch in range(hparams.epoch):

    train_epoch_loss = 0
    test_epoch_loss = 0

    model.train()
    for input in tqdm(train_dataloader):
        pred = model(input)
        loss = criterion(pred, input)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss

    model.eval()
    with torch.no_grad():
        for input in tqdm(test_dataloader):
            pred = model(input)
            loss = criterion(pred, input)

            test_epoch_loss += loss

    avg_train_loss = train_epoch_loss / len(train_dataloader)
    avg_test_loss = test_epoch_loss / len(test_dataloader)
