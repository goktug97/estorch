from typing import List
from functools import lru_cache

import numpy as np
import torch

@lru_cache(maxsize=1)
def center_function(population_size):
    centers = np.arange(0, population_size)
    centers = centers / (population_size - 1)
    centers -= 0.5
    return centers

def compute_ranks(fitness_scores):
    fitness_scores = np.array(fitness_scores)
    ranks = np.empty(fitness_scores.size, dtype=int)
    ranks[fitness_scores.argsort()] = np.arange(fitness_scores.size)
    return ranks

def rank_transformation(fitness_scores):
    ranks = compute_ranks(fitness_scores)
    population_size = len(fitness_scores)
    values = center_function(population_size)
    return values[ranks]

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

    def forward(self, policy):
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

    def forward(self, policy):
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
                total_reward += reward
        return total_reward, None

class ES():
    def __init__(self, policy, agent, optimizer, population_size, sigma=0.01,
                 policy_kwargs={}, agent_kwargs={}, optimizer_kwargs={}):
        self.policy = policy(**policy_kwargs)
        self.target = policy(**policy_kwargs)
        self.agent = agent(**agent_kwargs)
        self.optimizer = optimizer(self.policy.parameters(), **optimizer_kwargs)
        self.sigma = sigma
        self.population_size = population_size

        # FIX
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        import time
        while True:
            parameters = torch.nn.utils.parameters_to_vector(self.policy.parameters())
            normal = torch.distributions.normal.Normal(0, self.sigma)
            epsilon = normal.sample([self.population_size, parameters.shape[0]])
            parameters = parameters + epsilon
            rewards = []
            prev_time = time.time()
            for parameter in parameters:
                torch.nn.utils.vector_to_parameters(parameter, self.target.parameters())
                reward, _ = self.agent.forward(self.target)
                rewards.append(reward)
            ranked_rewards = torch.from_numpy(
                rank_transformation(rewards)
            ).unsqueeze(0).float().to(self.device)
            grad = (torch.mm(ranked_rewards, epsilon) /
                    (self.population_size * self.sigma)).squeeze()
            index = 0
            for parameter in self.policy.parameters():
                size = np.prod(parameter.shape)
                parameter.grad = -grad[index:index+size].view(parameter.shape)
                parameter.grad.data.clamp_(-1.0, 1.0)
                index += size
            self.optimizer.step()
            print(np.max(rewards))
            print(time.time() - prev_time)

class NS_ES():
    pass

class NSR_ES():
    pass

class NSRA_ES():
    pass

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

# bipedal = Bipedal()
# policy = Policy()
# reward, bc = bipedal.forward(policy)
# print(reward)
es = ES(CartPolePolicy, CartPole, torch.optim.Adam, 100)
# es = ES(Policy, Bipedal, torch.optim.Adam, 256)
es.train()


# a = torch.Tensor([1, 2, 3])
# print(a.data.detach().cpu().numpy())
# # Converting weights to an array
# ## One solution
# print(torch.cat([param.view(-1) for param in policy.parameters()]))
# 
# ## Another Solution
# print(torch.nn.utils.parameters_to_vector(policy.parameters()))
# 
# ## Going back
# parameters = torch.nn.utils.parameters_to_vector(policy.parameters())
# torch.nn.utils.vector_to_parameters(parameters, policy.parameters())
