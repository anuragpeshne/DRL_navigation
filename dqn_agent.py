#!/usr/bin/env python3

import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from cnn_model import CNN_QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
HUNGER_LIMIT = 20       # how many steps to tolerate wihtout +ve reward before doing something random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("selected device:", device)

class Agent():
    """Interacts with and leans from the environment."""

    def __init__(self, state_size, action_size, seed, input_type="vector"):
        """Initialize an Agent object.

        Params
        ======
           state_size (int): dimension of each state
           action_size (int): dimension of each action
           seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        if input_type == "vector":
            self.model_type = "fcnet"
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        else:
            self.model_type = "cnnet"
            self.qnetwork_local = CNN_QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = CNN_QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.model_type)
        # initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.hunger = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0., last_reward=1):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = self._preprocess_state(state)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        do_random = False

        # If we did not get reward from long time then
        # do something else, don't trust the network
        if (last_reward < 1):
            self.hunger += 1
            if self.hunger > HUNGER_LIMIT:
                do_random = True
                #print("Hunger limit crossed, executing random action!")
                self.hunger = 0
        else:
            self.hunger = 0

        # Epsilon-greedy action selection
        if not do_random and random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def _preprocess_state(self, state):
        if self.model_type == "fcnet":
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        else:
            rgb_transforms = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            state = torch.stack([rgb_transforms(state_sample) for state_sample in state]).float().to(device)
        return state

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets = self._get_targets(states, actions, rewards, next_states, dones, gamma)
        Q_expected = self.__get_expected(states, actions, rewards, next_states, dones)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def _get_targets(self, states, actions, rewards, next_states, dones, gamma):
        # Q0 = Q0 + alpha * (gamma * Q1 + reward - Q0)
        # qnet_local0 = qnet_local0 + alpha * (gamma * qnet_target1 + reward - qnet_local0)

        qnet_target1 = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        return rewards + (gamma * qnet_target1 * (1 - dones))

    def __get_expected(self, states, actions, rewards, next_states, dones):
        qnet_local0 = self.qnetwork_local(states).gather(1, actions)
        return qnet_local0

class DoubleDQNAgent(Agent):
    def _get_targets(self, states, actions, rewards, next_states, dones, gamma):
        # pick best actions using estimate network
        # evaluate the action and state using target network

        # torch.max returns max values as index 0 and indices as index 1
        best_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        qnet_target = self.qnetwork_target(next_states).detach().gather(1, best_actions)
        return rewards + (gamma * qnet_target * (1 - dones))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, model_type):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.model_type = model_type

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        if self.model_type == "fcnet":
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        else:
            rgb_transforms = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            states = torch.stack([rgb_transforms(e.state[0]) for e in experiences if e is not None]).float().to(device)
            next_states = torch.stack([rgb_transforms(e.next_state[0]) for e in experiences if e is not None]).float().to(device)

        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
