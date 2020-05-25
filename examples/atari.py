import collections
import random
import os

import torch
from torch import nn
from torch.nn import functional as F
import gym
import numpy as np
from skimage.transform import resize
from estorch import ES, VirtualBatchNorm
from mpi4py import MPI

class Policy(nn.Module):
    def __init__(self, n_actions, xref):
        super(Policy, self).__init__()
        self.xref = xref
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.bn1 = VirtualBatchNorm(16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.bn2 = VirtualBatchNorm(32)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        # First Pass for the Virtual Batch Normalization
        xref = F.relu(self.bn1(self.conv1(self.xref)))
        xref = F.relu(self.bn2(self.conv2(xref)))

        # Second Pass for the actual output
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 2592)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Agent():
    """The same preprocessing is applied to the frames as described in the paper
    `Human-level control through deep reinforcement learning`."""
    def __init__(self, env_name, frame_skip=4, device=torch.device('cpu')):
        self.env = gym.make(env_name)
        self.device = device
        self.frame_skip = frame_skip
        self.frames = collections.deque(maxlen=frame_skip)
        self.frame_buffer = collections.deque(maxlen=2)

    def rollout(self, policy, render=False):
        observation = self.env.reset()
        self.frame_buffer.append(observation)
        y_channel = observation @ np.array([0.299, 0.587, 0.114])
        y_channel = resize(y_channel, (84, 84, 1))
        for _ in range(self.frame_skip):
            self.frames.append(y_channel.transpose(2, 0, 1))
        done = False
        total_reward = 0
        with torch.no_grad():
            while not done:
                frame = torch.from_numpy(
                    np.concatenate(self.frames)).float().unsqueeze(0).to(device)
                action = policy(frame).squeeze().max(0)[1].item()
                for _ in range(frame_skip):
                    observation, reward, done, info = self.env.step(action)
                    total_reward += reward
                    if done: break
                    self.frame_buffer.append(observation)
                    observation = np.max((np.stack(self.frame_buffer)), axis=0)
                    y_channel = observation @ np.array([0.299, 0.587, 0.114])
                    y_channel = resize(y_channel, (84, 84, 1))
                    self.frames.append(y_channel.transpose(2, 0, 1))
                    if render:
                        self.env.render()
        return total_reward

if __name__ == '__main__':
    # Atari games take SO MUCH TIME to finish. SO MUCH.

    device = torch.device('cpu')
    n_proc = 1
    env_name='Breakout-v0'
    frame_skip = 4

    # Each MPI proccess is an "independent" program. Reference batch calculations
    # take some time and should not be calculated more than once. So we will
    # calculate it on the master process and broadcast it to other processes.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    reference_batch = []
    n_actions = 0
    if (rank == 0 and os.getenv('MPI_PARENT') is not None) or n_proc == 1:
        env = gym.make(env_name)
        for _ in range(5):
            observation = env.reset()
            frames = collections.deque(maxlen=frame_skip)
            frame_buffer = collections.deque(maxlen=2)
            frame_buffer.append(observation)
            y_channel = observation @ np.array([0.299, 0.587, 0.114])
            y_channel = resize(y_channel, (84, 84, 1))
            for _ in range(frame_skip):
                frames.append(y_channel.transpose(2, 0, 1))
            done = False
            total_reward = 0
            n_actions = env.action_space.n
            while not done:
                frame = torch.from_numpy(
                    np.concatenate(frames)).float().unsqueeze(0).to(device)
                reference_batch.append(frame)
                action = np.random.randint(0, n_actions)
                for _ in range(frame_skip):
                    observation, reward, done, info = env.step(action)
                    if done: break
                    frame_buffer.append(observation)
                    observation = np.max((np.stack(frame_buffer)), axis=0)
                    y_channel = observation @ np.array([0.299, 0.587, 0.114])
                    y_channel = resize(y_channel, (84, 84, 1))
                    frames.append(y_channel.transpose(2, 0, 1))
                    total_reward += reward
        reference_batch = random.sample(reference_batch, 128)
        reference_batch = torch.cat(reference_batch)
        comm.bcast(reference_batch, root=0)
        comm.bcast(n_actions, root=0)
    elif rank != 0:
        reference_batch = comm.bcast(reference_batch, root=0)
        n_actions = comm.bcast(n_actions, root=0)

    es = ES(policy=Policy, agent=Agent, optimizer=torch.optim.Adam,
            population_size=100, sigma=0.02, device=device,
            policy_kwargs={'n_actions': n_actions, 'xref': reference_batch},
            agent_kwargs={'env_name': env_name, 'frame_skip': frame_skip,
                          'device': device},
            optimizer_kwargs={'lr': 0.01})
    es.train(n_steps=100, n_proc=n_proc)

    # Latest Policy
    reward = agent.rollout(es.policy, render=True)
    print(f'Latest Policy Reward: {reward}')

    # Policy with the highest reward
    policy = Policy(n_actions, reference_batch).to(device)
    policy.load_state_dict(es.best_policy_dict)
    reward = agent.rollout(policy, render=True)
    print(f'Best Policy Reward: {reward}')
