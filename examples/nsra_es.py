import collections

import gym
import torch
import numpy as np
from estorch import NSRA_ES

class Agent():
    """NS-ES, NSR-ES, NSRA-ES algorithms require additional signal in addition to the
    reward signal. This signal is called behaviour characteristics and it is domain
    dependent signal which has to be chosen by the user. For more information look
    into references."""
    def __init__(self, device=torch.device('cpu'), n=32):
        self.env = gym.make('BipedalWalker-v3')
        self.device = device

        # In BipedalWalker, I've decided to make the
        # behaviour characterics as first
        # 32 actions and last 32 actions. In my tests,
        # with random actions, the shortest simulation was
        # 39 actions so I think 32 is a safe number.
        self.n = n

    def rollout(self, policy, render=False):
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
                if render:
                    self.env.render()
                total_reward += reward
                last_actions.append(action)
                if step < self.n:
                    start_actions.append(action)
                step+=1
        bc = np.concatenate([start_actions, last_actions]).flatten()
        return total_reward, bc

class Policy(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(Policy, self).__init__()
        self.linear_1 = torch.nn.Linear(n_input, 64)
        self.activation_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(64, 64)
        self.activation_2 = torch.nn.ReLU()
        self.linear_3 = torch.nn.Linear(64, n_output)

    def forward(self, x):
        l1 = self.linear_1(x)
        a1 = self.activation_1(l1)
        l2 = self.linear_2(a1)
        a2 = self.activation_2(l2)
        l3 = self.linear_3(a2)
        return l3


if __name__ == '__main__':
    # Usages for other novelty search algorithms are the same.
    # NS_ES ignores reward signals and optimizes purely for the novelty
    # NSR_ES uses novelty and reward signals equally.
    # NSRA_ES adaptively changes the importance of the signals.
    device = torch.device("cpu")
    agent = Agent()
    n_input = agent.env.observation_space.shape[0]
    n_output = agent.env.action_space.shape[0]
    es = NSRA_ES(Policy, Agent, torch.optim.Adam, population_size=256, sigma=0.02,
                 weight_t=10,
                 device=device, policy_kwargs={'n_input': n_input, 'n_output': n_output},
                 agent_kwargs={'device': device}, optimizer_kwargs={'lr': 0.01})

    es.train(n_steps=200, n_proc=2)

    # Meta Population Rewards
    for idx, (policy, _) in enumerate(es.meta_population):
        reward, bc = agent.rollout(policy, render=True)
        print(f'Reward of {idx}. policy from the meta population: {reward}')

    # Policy with the highest reward
    policy = Policy(n_input, n_output).to(device)
    policy.load_state_dict(es.best_policy_dict)
    reward, bc = agent.rollout(policy, render=True)
    print(f'Best Policy Reward: {reward}')
