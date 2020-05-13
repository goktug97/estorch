import os
import sys
import subprocess
from typing import List
from functools import lru_cache

from mpi4py import MPI
import numpy as np
import torch

class Tag():
    STOP = 1

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
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 policy_kwargs={}, agent_kwargs={}, optimizer_kwargs={}):

        self._comm = MPI.COMM_WORLD
        self.rank = self._comm.Get_rank()
        self.n_workers = self._comm.Get_size()

        self.population_size = population_size
        assert not (self.population_size % self.n_workers)

        self.device = device

        if self.rank == 0:
            self.policy = policy(**policy_kwargs).to(self.device)
            self.optimizer = optimizer(self.policy.parameters(), **optimizer_kwargs)
            self.sigma = sigma
            self._stop = False

        self.target = policy(**policy_kwargs).to(self.device)
        self.agent = agent(**agent_kwargs)
        parameters = torch.nn.utils.parameters_to_vector(self.target.parameters())
        self.n_parameters = parameters.shape[0]

        self.status = MPI.Status()
        self._trained = False

    def terminate(self):
        self._stop = True

    def log(self):
        """log function is called after every optimization step."""
        print(f'Step: {self.step} Episode Reward: {self.episode_reward}')

    def _master(self):
        self.step = 0
        with torch.no_grad():
            while self.step < self.n_steps and not self._stop:
                parameters = torch.nn.utils.parameters_to_vector(
                    self.policy.parameters())
                normal = torch.distributions.normal.Normal(0, self.sigma)
                epsilon = normal.sample([self.population_size, parameters.shape[0]])
                self.population_parameters = parameters + epsilon
                n_parameters_per_worker = int(self.population_size/self.n_workers)
                split_parameters = torch.split(self.population_parameters,
                                               n_parameters_per_worker)
                for i in range(1, self.n_workers):
                    self._comm.Send(split_parameters[i].numpy(), dest=i)

                self.population_rewards = np.empty(self.population_size)
                for idx, parameter in enumerate(split_parameters[0]):
                    torch.nn.utils.vector_to_parameters(
                        parameter, self.target.parameters())
                    reward, _ = self.agent.forward(self.target)
                    self.population_rewards[idx] = reward

                for worker_idx in range(1, self.n_workers):
                    self._comm.Recv(self.population_rewards[
                        worker_idx*n_parameters_per_worker:
                        worker_idx*n_parameters_per_worker+
                        n_parameters_per_worker],
                                    source=worker_idx)

                ranked_rewards = torch.from_numpy(
                    rank_transformation(self.population_rewards)).unsqueeze(0).float()
                grad = (torch.mm(ranked_rewards, epsilon) /
                        (self.population_size * self.sigma)).squeeze()
                index = 0
                for parameter in self.policy.parameters():
                    size = np.prod(parameter.shape)
                    parameter.grad = -grad[index:index+size].view(parameter.shape)
                    # Limit gradient update to increase stability.
                    parameter.grad.data.clamp_(-1.0, 1.0)
                    index += size
                self.optimizer.step()
                self.episode_reward, _ = self.agent.forward(self.policy)
                self.log()
                self.step += 1
        for worker_idx in range(1, self.n_workers):
            self._comm.send(None, dest=worker_idx, tag=Tag.STOP)

    def _slave(self):
        with torch.no_grad():
            while True:
                rewards = []
                parameters = np.empty((int(self.population_size/self.n_workers),
                                       self.n_parameters))
                self._comm.Recv(parameters, source=0, status=self.status)
                tag = self.status.Get_tag()
                if tag == Tag.STOP:
                    break
                parameters = torch.from_numpy(parameters).float()
                for parameter in parameters:
                    torch.nn.utils.vector_to_parameters(
                        parameter, self.target.parameters())
                    reward, _ = self.agent.forward(self.target)
                    rewards.append(reward)
                self._comm.Send(np.array(rewards), dest=0)
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


class NS_ES():
    pass

class NSR_ES():
    pass

class NSRA_ES():
    pass
