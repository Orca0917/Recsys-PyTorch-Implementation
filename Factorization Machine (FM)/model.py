import torch
import torch.nn as nn
import numpy as np


class FactorizationMachine(nn.Module):
    def __init__(self, field_dims, embedding_dim: int):
        super(FactorizationMachine, self).__init__()
        self.field_dims = field_dims
        self.embedding_dim = embedding_dim    # embedding dimension size

        # Embedding
        self.embedding = nn.Embedding(int(sum(field_dims)), embedding_dim)
        self.offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=int)

        # Linear (FC Layer)
        self.fc_layer = nn.Embedding(int(sum(field_dims)), 1)
        self.bias = nn.Parameter(torch.zeros((1, )))

        # initialization
        nn.init.xavier_uniform_(self.embedding.weight.data)
        nn.init.xavier_uniform_(self.fc_layer.weight.data)

    def forward(self, x: torch.tensor):
        offset_tensor = x + x.new_tensor(self.offset).unsqueeze(0)
        offset_tensor = offset_tensor.to(torch.long)

        # Linear term
        linear_term = torch.sum(self.fc_layer(offset_tensor), dim=1) + self.bias

        # Interaction term
        embedded_x = self.embedding(offset_tensor)
        square_of_sum = torch.sum(embedded_x, dim=1) ** 2
        sum_of_square = torch.sum(embedded_x ** 2, dim=1)
        interaction_term = 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)

        # output
        output = linear_term + interaction_term
        output = torch.sigmoid(output.squeeze(1))

        return output
