import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers: int = 64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers // 2)
        self.fc3 = nn.Linear(hidden_layers // 2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
