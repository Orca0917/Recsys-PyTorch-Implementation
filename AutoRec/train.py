import torch
import hparams
import torch.nn as nn
import torch.optim as optim

from model import AutoRec
from dataset import AutoRecDataset
from torch.utils.data import DataLoader


# load dataset
train_dataset = AutoRecDataset()
test_dataset = AutoRecDataset(is_train=False)

# make dataloader
train_dataloader = DataLoader(
    train_dataset, batch_size=hparams.batch_size, shuffle=True
)
test_dataloader = DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)

# model, optimizer definition
model = AutoRec(input_dim=hparams.input_dim, hidden_dim=hparams.hidden_dim).cuda()
optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
criterion = nn.MSELoss()

for epoch in range(hparams.epoch):

    train_epoch_loss = 0
    test_epoch_loss = 0

    # train
    model.train()
    for train_input in train_dataloader:
        # training with gpu
        train_input = train_input.cuda()

        # make prediction
        pred = model(train_input)

        # training only with observed data
        mask = train_input > 0
        loss = torch.sqrt(criterion(pred * mask, train_input * mask))

        # model optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss update
        train_epoch_loss += loss

    # test
    model.eval()
    with torch.no_grad():
        for test_input in test_dataloader:
            # testing with gpu
            test_input = test_input.cuda()

            # make prediction
            pred = model(test_input)

            # testing only with observed data
            mask = test_input > 0
            loss = torch.sqrt(criterion(pred * mask, test_input * mask))

            # loss update
            test_epoch_loss += loss

    # calculate average loss
    avg_train_loss = train_epoch_loss / len(train_dataloader)
    avg_test_loss = test_epoch_loss / len(test_dataloader)

    print("-" * 10)
    print(f"Epoch #{epoch}")
    print(f"Train loss : {avg_train_loss: .3f}")
    print(f"Test loss : {avg_test_loss: .3f}\n")
