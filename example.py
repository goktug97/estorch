
from estorch import ES
import numpy as np
import torch

class Policy(torch.nn.Module):
    def __init__(self):
        env = gym.make('BipedalWalker-v3')
        super(Policy, self).__init__()
        self.linear_1 = torch.nn.Linear(env.observation_space.shape[0], 64)
        self.activation_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(64, 64)
        self.activation_2 = torch.nn.ReLU()
        self.linear_3 = torch.nn.Linear(64, env.action_space.shape[0])

    def forward(self, x):
        l1 = self.linear_1(x)
        a1 = self.activation_1(l1)
        l2 = self.linear_2(a1)
        a2 = self.activation_2(l2)
        l3 = self.linear_3(a2)
        return l3

class CartPolePolicy(torch.nn.Module):
    def __init__(self):
        env = gym.make('CartPole-v1')
        super(CartPolePolicy, self).__init__()
        self.linear_1 = torch.nn.Linear(env.observation_space.shape[0], 64)
        self.activation_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(64, 64)
        self.activation_2 = torch.nn.ReLU()
        self.linear_3 = torch.nn.Linear(64, env.action_space.n)

    def forward(self, x):
        l1 = self.linear_1(x)
        a1 = self.activation_1(l1)
        l2 = self.linear_2(a1)
        a2 = self.activation_2(l2)
        l3 = self.linear_3(a2)
        return l3

import gym
import collections
class Bipedal():
    """This class will be passed to the ES algorithm.
    It should implement a forward function which
    returns either reward, behaviour characteristics
    or both depending on the chosen ES algorithm."""
    def __init__(self):
        self.env = gym.make('BipedalWalker-v3')

        # FIX
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Behaviour characteristic is domain dependent function,
        # in BipedalWalker, I've decided to make it first
        # 32 actions and last 32 actions. In my tests,
        # with random actions, the shortest simulation was
        # 39 actions so I think 32 is a safe number.
        self.n = 32

    def forward(self, policy, render=False):
        done = False
        observation = self.env.reset()
        step = 0
        total_reward = 0
        start_actions = []
        last_actions = collections.deque(maxlen=self.n)
        with torch.no_grad():
            while not done:
                observation = (torch.from_numpy(observation)
                               .float()
                               .to(self.device))
                action = (policy(observation)
                        .data
                        .detach()
                        .cpu()
                        .numpy())
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                last_actions.append(action)
                if step < self.n:
                    start_actions.append(action)
                step+=1
        bc = np.concatenate([start_actions, last_actions]).flatten()
        return total_reward, bc

class CartPole():
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, policy, render=False):
        done = False
        observation = self.env.reset()
        total_reward = 0
        with torch.no_grad():
            while not done:
                observation = (torch.from_numpy(observation)
                               .float()
                               .to(self.device))
                action = policy(observation).max(0)[1].item()
                observation, reward, done, info = self.env.step(action)
                if render:
                    self.env.render()
                total_reward += reward
        return total_reward, None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# es = ES(Policy, Bipedal, torch.optim.Adam, population_size=256, device=device)
es = ES(CartPolePolicy, CartPole, torch.optim.Adam, population_size=100,
        device=device)
es.train(n_steps=100, n_proc=2)
es.agent.forward(es.policy, render=True)
