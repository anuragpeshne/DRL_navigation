#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        hidden_layer_size1 = 64
        hidden_layer_size2 = 64
        
        # 84x84x3 -> CNN1 -> 8_x8_x3 -> CNN2 -> 7_x7_x3 -> FC1 -> FC2
        
        self.conv1 = nn.Conv2d(state_size, hidden_layer_size1, kernel=3)
        

        self.fc1 = nn.Linear(state_size, hidden_layer_size1)
        self.fc2 = nn.Linear(hidden_layer_size1, hidden_layer_size2)
        self.fc3 = nn.Linear(hidden_layer_size2, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x