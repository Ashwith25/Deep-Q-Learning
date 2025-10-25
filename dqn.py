import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, states, h1, h2, actions):
        super().__init__()
        self.fc1 = nn.Linear(states, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
