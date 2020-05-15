import os
import sys
import subprocess
from typing import List
from functools import lru_cache
from enum import Enum

from scipy import spatial
from mpi4py import MPI
import numpy as np
import torch

class Tag():
    STOP = 1

class Algorithm(Enum):
    classic = 1
    novelty = 2

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
            command = f"mpirun{' --bind-to hwthread ' if hwthread else ' '}-np {n_proc}"
            command = f'{command} {sys.executable} -u {os.path.abspath(module.__file__)}'
        subprocess.call(command.split(' '), env=env)
        return True
    return False

class ES():
    _ALGORITHM_TYPE = Algorithm.classic
    def __init__(self, policy, agent, optimizer, population_size, sigma=0.01,
                 device=torch.device("cpu"),
                 policy_kwargs={}, agent_kwargs={}, optimizer_kwargs={}):

        self._comm = MPI.COMM_WORLD
        self.rank = self._comm.Get_rank()
        self.n_workers = self._comm.Get_size()

        self.population_size = population_size
        assert not (self.population_size % self.n_workers)

        self.device = device

        if self.rank == 0:
            if self._ALGORITHM_TYPE == Algorithm.classic:
                self.policy = policy(**policy_kwargs).to(self.device)
                self.optimizer = optimizer(self.policy.parameters(), **optimizer_kwargs)
            self.sigma = sigma
            self._stop = False

        self.agent = agent(**agent_kwargs)
        self.target = policy(**policy_kwargs).to(self.device)
        parameters = torch.nn.utils.parameters_to_vector(self.target.parameters())
        self.n_parameters = parameters.shape[0]
        self.best_reward = -float('inf')

        self.status = MPI.Status()
        self._trained = False

    def terminate(self):
        self._stop = True

    def log(self):
        """log function is called after every optimization step."""
        print(f'Step: {self.step}')
        print(f'Episode Reward: {self.episode_reward}')
        print(f'Max Population Reward: {np.max(self.population_returns)}')

    def _calculate_grad(self, epsilon):
        ranked_rewards = torch.from_numpy(
            rank_transformation(self.population_returns)).unsqueeze(0).float()
        grad = (torch.mm(ranked_rewards, epsilon) /
                (self.population_size * self.sigma)).squeeze()
        return grad

    def _after_optimize(self, policy):
        self.episode_reward, _ = self.agent.rollout(policy)
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self.best_policy_dict = policy.state_dict()

    def _master(self):
        self.step = 0
        with torch.no_grad():
            while self.step < self.n_steps and not self._stop:
                parameters = torch.nn.utils.parameters_to_vector(
                    self.policy.parameters())
                normal = torch.distributions.normal.Normal(0, self.sigma)
                epsilon = normal.sample([self.population_size, parameters.shape[0]])
                self.population_parameters = parameters.detach().cpu() + epsilon
                n_parameters_per_worker = int(self.population_size/self.n_workers)
                split_parameters = torch.split(self.population_parameters,
                                               n_parameters_per_worker)
                for i in range(1, self.n_workers):
                    self._comm.Send(split_parameters[i].numpy(), dest=i)

                self.population_returns = np.empty(self.population_size)
                for idx, parameter in enumerate(split_parameters[0]):
                    torch.nn.utils.vector_to_parameters(
                        parameter.to(self.device), self.target.parameters())
                    reward, _ = self.agent.rollout(self.target)
                    self.population_returns[idx] = reward

                for worker_idx in range(1, self.n_workers):
                    self._comm.Recv(self.population_returns[
                        worker_idx*n_parameters_per_worker:
                        worker_idx*n_parameters_per_worker+
                        n_parameters_per_worker],
                                    source=worker_idx)

                grad = self._calculate_grad(epsilon)
                index = 0
                for parameter in self.policy.parameters():
                    size = np.prod(parameter.shape)
                    parameter.grad = (-grad[index:index+size]
                                      .view(parameter.shape)
                                      .to(self.device))
                    # Limit gradient update to increase stability.
                    parameter.grad.data.clamp_(-1.0, 1.0)
                    index += size
                self.optimizer.step()
                self._after_optimize(self.policy)
                self.log()
                self.step += 1
        for worker_idx in range(1, self.n_workers):
            self._comm.send(None, dest=worker_idx, tag=Tag.STOP)

    def _slave(self):
        with torch.no_grad():
            while True:
                returns = []
                parameters = np.empty((int(self.population_size/self.n_workers),
                                       self.n_parameters), dtype=np.float32)
                self._comm.Recv(parameters, source=0, status=self.status)
                tag = self.status.Get_tag()
                if tag == Tag.STOP:
                    break
                parameters = torch.from_numpy(parameters).float()
                for parameter in parameters:
                    torch.nn.utils.vector_to_parameters(
                        parameter.to(self.device), self.target.parameters())
                    reward, _ = self.agent.rollout(self.target)
                    returns.append(reward)
                self._comm.Send(np.array(returns), dest=0)
            sys.exit(0)

    def train(self, n_steps, n_proc=1, hwthread=False, hostfile=None):
        if self._trained:
            error_message = "train function can not be called more than once."
            error_message = f"\033[1m\x1b[31m{error_message}\x1b[0m\x1b[0m"
            raise RuntimeError(error_message)
        self._trained = True
        self.n_steps = n_steps
        if fork(n_proc, hwthread, hostfile): sys.exit(0)
        self._master() if self.rank == 0 else self._slave()

class NS_ES(ES):
    _ALGORITHM_TYPE = Algorithm.novelty
    def __init__(self, policy, agent, optimizer, population_size, sigma=0.01,
                 meta_population_size=3, k=10, device=torch.device("cpu"),
                 policy_kwargs={}, agent_kwargs={}, optimizer_kwargs={}):

        super().__init__(policy, agent, optimizer, population_size, sigma,
                         device, policy_kwargs, agent_kwargs, optimizer_kwargs)

        self.meta_population_size = meta_population_size
        self.k = k

        if self.rank == 0:
            self.archive = []
            self.meta_population = []
            for _ in range(self.meta_population_size):
                p = policy(**policy_kwargs).to(self.device)
                optim = optimizer(p.parameters(), **optimizer_kwargs)
                self.meta_population.append((p, optim))
                reward, bc = self.agent.rollout(p)
                if bc is None:
                    raise ValueError("Behaviour Charateristics is None")
                self.archive.append(bc)

    def _calculate_novelty(self, bc, archive):
        kd = spatial.cKDTree(archive)
        distances, idxs = kd.query(bc, k=self.k)
        distances = distances[distances < float('inf')]
        novelty = np.sum(distances) / np.linalg.norm(archive)
        return novelty

    def _calculate_grad(self, epsilon):
        ranked_novelties = torch.from_numpy(
            rank_transformation(
                self.population_returns[:, 1])).unsqueeze(0).float()
        grad = (torch.mm(ranked_novelties, epsilon) /
                (self.population_size * self.sigma)).squeeze()
        return grad

    def _after_optimize(self, policy):
        self.episode_reward, bc = self.agent.rollout(policy)
        self.archive.append(bc)
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self.best_policy_dict = policy.state_dict()

    def _master(self):
        self.step = 0
        with torch.no_grad():
            while self.step < self.n_steps and not self._stop:
                total_novelty = []
                for policy, _ in self.meta_population:
                    reward, bc = self.agent.rollout(policy)
                    novelty = self._calculate_novelty(bc, self.archive)
                    total_novelty.append(novelty)
                total_novelty = np.array(total_novelty)
                meta_population_probability = total_novelty / np.sum(total_novelty)
                self.idx = np.random.choice(
                    np.arange(len(self.meta_population), dtype=np.int),
                    p=meta_population_probability)
                policy, optimizer = self.meta_population[self.idx]

                parameters = torch.nn.utils.parameters_to_vector(
                    policy.parameters())
                normal = torch.distributions.normal.Normal(0, self.sigma)
                epsilon = normal.sample([self.population_size, parameters.shape[0]])
                self.population_parameters = parameters.detach().cpu() + epsilon
                n_parameters_per_worker = int(self.population_size/self.n_workers)
                split_parameters = torch.split(self.population_parameters,
                                               n_parameters_per_worker)
                for i in range(1, self.n_workers):
                    self._comm.Send(split_parameters[i].numpy(), dest=i)
                self._comm.bcast(self.archive, root=0)

                self.population_returns = np.empty((self.population_size, 2))
                for idx, parameter in enumerate(split_parameters[0]):
                    torch.nn.utils.vector_to_parameters(
                        parameter.to(self.device), self.target.parameters())
                    reward, bc = self.agent.rollout(self.target)
                    novelty = self._calculate_novelty(bc, self.archive)
                    self.population_returns[idx] = reward, novelty

                for worker_idx in range(1, self.n_workers):
                    self._comm.Recv(self.population_returns[
                        worker_idx*n_parameters_per_worker:
                        worker_idx*n_parameters_per_worker+
                        n_parameters_per_worker],
                                    source=worker_idx)

                grad = self._calculate_grad(epsilon)
                index = 0
                for parameter in policy.parameters():
                    size = np.prod(parameter.shape)
                    parameter.grad = (-grad[index:index+size]
                                      .view(parameter.shape)
                                      .to(self.device))
                    # Limit gradient update to increase stability.
                    parameter.grad.data.clamp_(-1.0, 1.0)
                    index += size
                optimizer.step()
                self._after_optimize(policy)
                self.log()
                self.step += 1
        for worker_idx in range(1, self.n_workers):
            self._comm.send(None, dest=worker_idx, tag=Tag.STOP)

    def _slave(self):
        with torch.no_grad():
            while True:
                returns = []
                parameters = np.empty((int(self.population_size/self.n_workers),
                                       self.n_parameters), dtype=np.float32)
                self._comm.Recv(parameters, source=0, status=self.status)
                tag = self.status.Get_tag()
                if tag == Tag.STOP:
                    break
                archive = None
                archive = self._comm.bcast(archive, root=0)
                parameters = torch.from_numpy(parameters).float()
                for parameter in parameters:
                    torch.nn.utils.vector_to_parameters(
                        parameter.to(self.device), self.target.parameters())
                    reward, bc = self.agent.rollout(self.target)
                    novelty = self._calculate_novelty(bc, archive)
                    returns.append((reward, novelty))
                self._comm.Send(np.array(returns), dest=0)
            sys.exit(0)

class NSR_ES(NS_ES):
    _ALGORITHM_TYPE = Algorithm.novelty
    def _calculate_grad(self, epsilon):
        ranked_rewards = torch.from_numpy(
            rank_transformation(self.population_returns[:, 0])).unsqueeze(0).float()
        ranked_novelties = torch.from_numpy(rank_transformation(
                self.population_returns[:, 1])).unsqueeze(0).float()
        grad = (torch.mm((ranked_novelties+ranked_rewards)/2, epsilon) /
                (self.population_size * self.sigma)).squeeze()
        return grad

class NSRA_ES(NS_ES):
    _ALGORITHM_TYPE = Algorithm.novelty
    def __init__(self, policy, agent, optimizer, population_size, sigma=0.01, 
                 meta_population_size=3, k=10, min_weight=0.0, weight_t=50,
                 weight_delta=0.05, device=torch.device("cpu"),
                 policy_kwargs={}, agent_kwargs={}, optimizer_kwargs={}):
        super().__init__(policy=policy, agent=agent, optimizer=optimizer,
                         population_size=population_size, sigma=sigma,
                         meta_population_size=meta_population_size, k=k,
                         device=device, policy_kwargs=policy_kwargs,
                         agent_kwargs=agent_kwargs, optimizer_kwargs=optimizer_kwargs)

        if self.rank == 0:
            self.weight = 1.0
            self.min_weight = min_weight
            self.weight_t = weight_t
            self.weight_delta = 0.05
            self.t = 0

    def _calculate_grad(self, epsilon):
        ranked_rewards = torch.from_numpy(
            rank_transformation(self.population_returns[:, 0])).unsqueeze(0).float()
        ranked_novelties = torch.from_numpy(
            rank_transformation(self.population_returns[:, 1])).unsqueeze(0).float()
        grad = (torch.mm(self.weight*ranked_rewards+
                         (1.0-self.weight)*ranked_novelties, epsilon) /
                (self.population_size * self.sigma)).squeeze()
        return grad

    def _after_optimize(self, policy):
        self.episode_reward, bc = self.agent.rollout(policy)
        self.archive.append(bc)
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self.weight = min(self.weight + self.weight_delta, 1.0)
            self.best_policy_dict = policy.state_dict()
            self.t = 0
        else:
            self.t += 1
            if self.t >= self.weight_t:
                self.weight = max(self.weight - self.weight_delta, self.min_weight)
