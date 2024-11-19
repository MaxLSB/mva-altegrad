"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    """Simple GNN model"""

    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13

        x = self.relu(
            torch.spmm(adj, torch.mm(x_in, self.fc1.weight.t()) + self.fc1.bias)
        )
        x = self.dropout(x)
        x = self.relu(torch.spmm(adj, torch.mm(x, self.fc2.weight.t()) + self.fc2.bias))
        x_hidden = x
        x = torch.mm(x, self.fc3.weight.t()) + self.fc3.bias

        return F.log_softmax(x, dim=1), x_hidden
