import torch.nn as nn


class AutoRec(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=4):
        super(AutoRec, self).__init__()
        self.input_dim = input_dim
        self.num_hidden = hidden_dim

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.decoder(x)

        return x
