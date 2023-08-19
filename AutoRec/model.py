import torch
import torch.nn as nn


class AutoRec(nn.Module):
    def __init__(self, input_dim=9724, hidden_dim=500):
        super(AutoRec, self).__init__()
        self.input_dim = input_dim
        self.num_hidden = hidden_dim

        self.encoder = nn.Linear(input_dim, hidden_dim, dtype=torch.float64)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.decoder = nn.Linear(hidden_dim, input_dim, dtype=torch.float64)

    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.decoder(x)

        return x
