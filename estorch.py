import os
import sys
import subprocess
from typing import List
from functools import lru_cache

from mpi4py import MPI
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

def fork(n_proc=1, hwthread=False, hostfile=None):
    if os.getenv('MPI_PARENT') is None:
        import inspect
        frame = inspect.stack()[2]
        module = inspect.getmodule(frame[0])
        env = os.environ.copy()
        env['MPI_PARENT'] = '1'
        if hostfile:
            command = f"mpirun --hostfile {hostfile}"
            command = f"{command} {sys.executable} -u {os.path.abspath(module.__file__)}"
        else:
            command = f"mpirun{' -use-hwthread-cpus ' if hwthread else ' '}-np {n_proc}"
            command = f'{command} {sys.executable} -u {os.path.abspath(module.__file__)}'
        subprocess.call(command.split(' '), env=env)
        return True
    return False

class ES():
    def __init__(self, policy, agent, optimizer, population_size, sigma=0.01,
                 policy_kwargs={}, agent_kwargs={}, optimizer_kwargs={}):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_workers = self.comm.Get_size() 

        if self.rank == 0:
            self.policy = policy(**policy_kwargs)
            self.optimizer = optimizer(self.policy.parameters(), **optimizer_kwargs)
            self.sigma = sigma

        self.target = policy(**policy_kwargs)
        self.agent = agent(**agent_kwargs)
        self.population_size = population_size
        assert not (self.population_size % self.n_workers)

    def master(self):
        step = 0
        while step < self.n_steps:
            with torch.no_grad():
                parameters = torch.nn.utils.parameters_to_vector(
                    self.policy.parameters())
                normal = torch.distributions.normal.Normal(0, self.sigma)
                epsilon = normal.sample([self.population_size, parameters.shape[0]])
                parameters = parameters + epsilon
                n_parameters_per_worker = int(self.population_size/self.n_workers)
                parameters = torch.split(parameters, n_parameters_per_worker)
                for i in range(1, self.n_workers):
                    self.comm.Send(parameters[i].numpy(), dest=i)

                rewards = np.empty(self.population_size)
                for idx, parameter in enumerate(parameters[0]):
                    torch.nn.utils.vector_to_parameters(
                        parameter, self.target.parameters())
                    reward, _ = self.agent.forward(self.target)
                    rewards[idx] = reward

                for worker_idx in range(1, self.n_workers):
                    self.comm.Recv(rewards[worker_idx*n_parameters_per_worker:
                                           worker_idx*n_parameters_per_worker+
                                           n_parameters_per_worker],
                                   source=worker_idx)

                ranked_rewards = torch.from_numpy(
                    rank_transformation(rewards)
                ).unsqueeze(0).float()
                grad = (torch.mm(ranked_rewards, epsilon) /
                        (self.population_size * self.sigma)).squeeze()
                index = 0
                for parameter in self.policy.parameters():
                    size = np.prod(parameter.shape)
                    parameter.grad = -grad[index:index+size].view(parameter.shape)
                    parameter.grad.data.clamp_(-1.0, 1.0)
                    index += size
                self.optimizer.step()
                step += 1
                print(np.max(rewards))

    def slave(self):
        step = 0
        while step < self.n_steps:
            rewards = []
            parameters = np.empty((int(self.population_size/self.n_workers),
                                   self.n_parameters))
            self.comm.Recv(parameters, source=0)
            parameters = torch.from_numpy(parameters).float()
            with torch.no_grad():
                for parameter in parameters:
                    torch.nn.utils.vector_to_parameters(
                        parameter, self.target.parameters())
                    reward, _ = self.agent.forward(self.target)
                    rewards.append(reward)
            self.comm.Send(np.array(rewards), dest=0)
            step += 1

    def train(self, n_steps, n_proc=1, hwthread=False, hostfile=None):
        self.n_steps = n_steps
        if fork(n_proc, hwthread, hostfile): sys.exit(0)
        parameters = torch.nn.utils.parameters_to_vector(self.target.parameters())
        self.n_parameters = parameters.shape[0]
        self.master() if self.rank == 0 else self.slave()


class NS_ES():
    pass

class NSR_ES():
    pass

class NSRA_ES():
    pass

