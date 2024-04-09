import numpy as np, torch, torch.nn as nn, torch.optim as optim
from utils import *
import torch.nn.functional as F


class Environment:

    def __init__ (self, X, k):

        self.X = X
        self.k = k

    def step(self, C, a = 1):

        assignment = torch.argmin(torch.cdist(self.X, C), axis=1)
        loss = torch.cdist(self.X, C)[torch.arange(len(self.X)), assignment].sum() / len(self.X)

        return loss


class Agent:

    def __init__ (self, X, k):

        self.X = X
        self.k = k
        self.device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
        self.input_dim = self.X.numel()


        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim//2),
            nn.ReLU(),
            nn.Linear(self.input_dim//2, self.input_dim//2),
            nn.ReLU(),
            nn.Linear(self.input_dim//2, k*len(self.X[0])),
        ).to(self.device)


    def select_action(self, state):
        return self.model(state).view(self.k, len(self.X[0]))
    

    def update_policy(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(env, agent, episodes=3, epsilon_start=0.9, epsilon_end=0.0, epochs=100):

    for episode in range(episodes):

        state = env.X.flatten()

        epsilon = epsilon_start - (epsilon_start - epsilon_end) * episode / (episodes - 1)

        optimizer = optim.Adam(agent.model.parameters(), lr=0.1)

        # Exploration phase
        for _ in range(epochs):

            if np.random.rand() < epsilon:
                with torch.no_grad():
                    random_indices = torch.randperm(env.X.shape[0])[:env.k]
                    action = env.X[random_indices].requires_grad_(True)
            elif np.random.rand() < epsilon:
                action = torch.mean(env.X, axis=0).requires_grad_(True)
                action = torch.stack([action] * env.k)
            else:
                action = agent.select_action(state)

            loss = env.step(action)
            agent.update_policy(optimizer, loss)

        print('Episode: ', episode, 'Loss: ', loss.item())