import torch.nn as nn
import torch.nn.functional as F


class VDNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers: int = 64):
        super(VDNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers // 2)
        self.fc3 = nn.Linear(hidden_layers // 2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def clip_gradients(self, clip_val: float):
        """ Clip gradients during backward pass """
        for param in self.parameters():
            param.register_hook(lambda grad: grad.clamp_(-clip_val, clip_val))