#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(CNN_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        hidden_layer_size1 = 64

        # 3 input image channel, 32 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (84-3)/1 +1 = 82
        # the output Tensor for one image, will have the dimensions: (32, 82, 82)
        # after one pool layer, this becomes (32, 41, 41)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)

        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (41-3)/1 +1 = 39
        # the output tensor will have dimensions: (64, 39, 39)
        # after another pool layer this becomes (64, 19, 19); 19.5 taking floor
        self.conv2 = nn.Conv2d(32, 64, 3)

        # third conv layer: 32 inputs, 32 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (19-3)/1 +1 = 17
        # the output tensor will have dimensions: (32, 17, 17)
        # after another pool layer this becomes (32, 8, 8); (floor(8.5))
        self.conv3 = nn.Conv2d(64, 32, 3)

        # 32 output * 8 * 8 pooled map size
        self.fc1 = nn.Linear(32 * 8 * 8, hidden_layer_size1)

        self.fc2 = nn.Linear(hidden_layer_size1, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # 2 conv networks
        #print(state.shape)
        #print(self.conv1(state).shape)
        #print(self.pool(F.relu(self.conv1(state))).shape)
        x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # prep for linear layer
        # Flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
