import torch
import hparams
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dataset import FMDataset
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
criterion = nn.MSELoss()

# Training
for epoch in range(hparams.num_epochs):
    epoch_loss = 0.0

    tqdm_train = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch #{epoch + 1}")
    for X, y in tqdm_train:
        output = model(X)
        y = y.to(torch.float32)
        loss = torch.sqrt(criterion(output, y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch #{epoch + 1} finished. Loss : {epoch_loss / len(train_dataloader): .4f}")

print("Training Factorization Machine terminated !")
