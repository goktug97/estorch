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


@lru_cache(maxsize=1)
def _center_function(population_size):
    centers = np.arange(0, population_size)
    centers = centers / (population_size - 1)
    centers -= 0.5
    return centers

def _compute_ranks(rewards):
    rewards = np.array(rewards)
    ranks = np.empty(rewards.size, dtype=int)
    ranks[rewards.argsort()] = np.arange(rewards.size)
    return ranks

def rank_transformation(rewards):
    r"""Applies rank transformation to the returns.

    Examples:
        >>> rewards = [-123, -50, 3, -5, 20, 10, 100]
        >>> estorch.rank_transformation(rewards)
        array([-0.5       , -0.33333333,  0.        , -0.16666667,  0.33333333,
                0.16666667,  0.5       ])
    """
    ranks = _compute_ranks(rewards)
    values = _center_function(len(rewards))
    return values[ranks]

def _fork(n_proc=1, hwthread=False, hostfile=None):
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


class _Tag():
    STOP = 1


class _Algorithm(Enum):
    classic = 1
    novelty = 2


class ES():
    """Classic Evolution Strategy Algorithm. It optimizes given
    policy for the max reward return. For example usage refer to
    https://github.com/goktug97/estorch/blob/master/examples/cartpole_es.py

    .. math::

        \\nabla_{\\theta} \\mathbb{E}_{\\epsilon \\sim N(0, I)} F(\\theta+\\sigma \\epsilon)=\\frac{1}{\\sigma} \\mathbb{E}_{\\epsilon \\sim N(0, I)}\{F(\\theta+\\sigma \\epsilon) \\epsilon\}
    
    - Evolution Strategies as a Scalable Alternative to Reinforcement Learning:
      https://arxiv.org/abs/1703.03864

    Args:
        policy: PyTorch Module. Should be passed as a ``class``.
        agent: Policy will be optimized to maximize the output of this
               class's rollout function. For an example agent class refer to;
               https://github.com/goktug97/estorch/blob/master/examples/cartpole_es.py
               Should be passed as a ``class``.
        optimizer: Optimizer that will be used to update parameters of the policy.
                   Any PyTorch optimizer can be used. Should be passed as a ``class``.
        population_size: Population size of the evolution strategy.

            .. note ::

                if you are using multiprocessing make sure ``population_size`` is
                multiple of ``n_proc``
        sigma: Standart Deviation to use while sampling the generation from the policy.
        device: Torch device

            .. note ::

                For every process a target network will be created to use during rollout.
                That is why I don't recommend use of ``torch.device('cuda')``.
        policy_kwargs: This dictionary of arguments will passed to the policy during 
                       initialization.
        agent_kwargs: This dictionary of arguments will passed to the agent during 
                      initialization.
        optimizer_kwargs: This dictionary of arguments will passed to
                          the optimizer during initialization.
                         
    :var policy: Each step this policy is optimized. Only in master process.
    :var optimizer: Optimizer that is used to optimize the
                    :attr:`policy`. Only in master process.
    :var agent: Used for rollout in each processes.
    :var n_parameters: Number of trainable parameters of the :attr:`policy`.
    :var best_reward: Best reward achived during the training.
    :var episode_reward: Reward of the policy after the optimization.
    :var best_policy_dict: PyTorch ``state_dict`` of the policy with the highest reward.
    :var population_returns: Current population's rewards.
    :var population_parameters: Parameter vectors of the current population.
    """

    _ALGORITHM_TYPE = _Algorithm.classic
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
            if self._ALGORITHM_TYPE == _Algorithm.classic:
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
        """Terminates the training and sends terminate signal to other processes."""
        self._stop = True

    def log(self):
        """``log`` function is called after every optimization step.
        This function can be used to interract with the model during the training.
        By default its contents are:
        
        .. code-block:: python

            print(f'Step: {self.step}')
            print(f'Episode Reward: {self.episode_reward}')
            print(f'Max Population Reward: {np.max(self.population_returns)}')
            print(f'Max Reward: {self.best_reward}')
        
        For example usage;
        https://github.com/goktug97/estorch/blob/master/examples/early_stopping.py
        """
        print(f'Step: {self.step}')
        print(f'Episode Reward: {self.episode_reward}')
        print(f'Max Population Reward: {np.max(self.population_returns)}')
        print(f'Max Reward: {self.best_reward}')

    def _calculate_grad(self, epsilon):
        ranked_rewards = torch.from_numpy(
            rank_transformation(self.population_returns.squeeze())).unsqueeze(0).float()
        grad = (torch.mm(ranked_rewards, epsilon) /
                (self.population_size * self.sigma)).squeeze()
        return grad

    def _after_optimize(self, policy):
        self.episode_reward = self.agent.rollout(policy)
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self.best_policy_dict = policy.state_dict()

    def _sample_policy(self, policy):
        parameters = torch.nn.utils.parameters_to_vector(policy.parameters())
        normal = torch.distributions.normal.Normal(0, self.sigma)
        epsilon = normal.sample([int(self.population_size/2), parameters.shape[0]])
        parameters = parameters.detach().cpu()
        population_parameters = torch.cat((parameters + epsilon, parameters - epsilon))
        return population_parameters, torch.cat((epsilon, -epsilon))

    def _calculate_returns(self, parameters):
        returns = []
        for parameter in parameters:
            torch.nn.utils.vector_to_parameters(
                parameter.to(self.device), self.target.parameters())
            reward = self.agent.rollout(self.target)
            returns.append(reward)
        return np.array(returns, dtype=np.float32)[:, np.newaxis]

    def _get_policy(self):
        return self.policy, self.optimizer

    def _send_to_slaves(self, split_parameters):
        for i in range(1, self.n_workers):
            self._comm.Send(split_parameters[i].numpy(), dest=i)

    def _master(self):
        self.step = 0
        with torch.no_grad():
            while self.step < self.n_steps and not self._stop:
                policy, optimizer = self._get_policy()
                self.population_parameters, epsilon = self._sample_policy(policy)
                n_parameters_per_worker = int(self.population_size/self.n_workers)
                split_parameters = torch.split(self.population_parameters,
                                               n_parameters_per_worker)

                self._send_to_slaves(split_parameters)

                returns = self._calculate_returns(split_parameters[0])
                self.population_returns = np.empty((
                    self.population_size, returns.shape[1]), dtype=np.float32)
                self.population_returns[:n_parameters_per_worker] = returns

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
            self._comm.send(None, dest=worker_idx, tag=_Tag.STOP)

    def _recv_from_master(self):
        parameters = np.empty((int(self.population_size/self.n_workers),
                               self.n_parameters), dtype=np.float32)
        self._comm.Recv(parameters, source=0, status=self.status)
        tag = self.status.Get_tag()
        if tag == _Tag.STOP:
            return
        parameters = torch.from_numpy(parameters).float()
        return parameters

    def _slave(self):
        with torch.no_grad():
            while True:
                parameters = self._recv_from_master()
                if parameters is None:
                    break
                returns = self._calculate_returns(parameters)
                self._comm.Send(returns, dest=0)
            sys.exit(0)

    def train(self, n_steps, n_proc=1, hwthread=False, hostfile=None):
        r"""Train Evolution Strategy algorithm for n_steps in n_proc processes.

        .. note::
            This function can not be called more than once in the same
            script if ``n_proc`` is set to more than 1 because it
            executes the same script ``n_proc`` times which means it
            will start from the beginning of the script everytime.
        
        Args:
            n_steps: Number of training steps.
            n_proc: Number of processes. Processes are used for rollouts.
            hwthread: A boolean value, if ``True`` use hardware
                      threads as independent cpus. Some processors are
                      hyperthreaded which means 1 CPU core is splitted into
                      multiple threads. For example in Linux, `nproc` command
                      returns number of cores and if that number doesn't work
                      here set hwthread to ``True`` and try again.
            hostfile: If set, ``n_proc`` and ``hwthread`` will be ignored and the
                      ``hostfile`` will be used to initialize
                      multiprocessing. For more information visit
                      https://github.com/open-mpi/ompi/blob/9c0a2bb2d675583934efd5e6e22ce8245dd5554c/README#L1904

        Raises:
            RuntimeError: train function can not be called more than once.
        """
        self.n_steps = n_steps
        if n_proc > 1:
            if self._trained:
                error_message = "train function can not be called more than once."
                error_message = f"\033[1m\x1b[31m{error_message}\x1b[0m\x1b[0m"
                raise RuntimeError(error_message)
            self._trained = True
            if _fork(n_proc, hwthread, hostfile): sys.exit(0)
            self._master() if self.rank == 0 else self._slave()
        else:
            self._master()


class NS_ES(ES):
    """Novelty Search Evolution Strategy Algorithm. It optimizes given
    policy for the max novelty return. For example usage refer to
    https://github.com/goktug97/estorch/blob/master/examples/nsra_es.py
    This class is inherited from the :class:`ES` so every function that is described
    in the :class:`ES` can be used in this class too.

    .. math::

        \\nabla_{\\theta_{t}} \\mathbb{E}_{e \\sim N(0, I)}\\left[N\\left(\\theta_{t}+\\sigma \\epsilon, A\\right) | A\\right] \\approx \\frac{1}{n \\sigma} \\sum_{i=1}^{n} N\\left(\\theta_{t}^{i}, A\\right) \\epsilon_{i}    
    
    Where :math:`N\\left(\\theta_{t}^{i}, A\\right)` is calculated as;

    .. math ::
        N(\\theta, A)=N\\left(b\\left(\\pi_{\\theta}\\right), A\\right)=\\frac{1}{|S|} \\sum_{j \\in S}\\left\|b\\left(\\pi_{\\theta}\\right)-b\\left(\\pi_{j}\\right)\\right\|_{2}

    .. math ::
        S=k N N\\left(b\\left(\\pi_{\\theta}\\right), A\\right)

    .. math ::
        =\\left\{b\\left(\\pi_{1}\\right), b\\left(\\pi_{2}\\right), \\ldots, b\\left(\\pi_{k}\\right)\\right\}
    

    - Improving Exploration in Evolution Strategies for Deep
      Reinforcement Learning via a Population of Novelty-Seeking Agents
      http://papers.nips.cc/paper/7750-improving-exploration-in-evolution-strategies-for-deep-reinforcement-learning-via-a-population-of-novelty-seeking-agents.pdf

    Args:
        policy: PyTorch Module. Should be passed as a ``class``.
        agent: Policy will be optimized to maximize the output of this
               class's rollout function. For an example agent class refer to;
               https://github.com/goktug97/estorch/blob/master/examples/cartpole_es.py
               Should be passed as a ``class``.
        optimizer: Optimizer that will be used to update parameters of the policy.
                   Any PyTorch optimizer can be used. Should be passed as a ``class``.
        population_size: Population size of the evolution strategy.

            .. note ::

                if you are using multiprocessing make sure ``population_size`` is
                multiple of ``n_proc``
        sigma: Standart Deviation to use while sampling the generation from the policy.
        meta_population_size: Instead of one policy a meta population
          of policies are optimized during
          training. Each step a policy is chosen
          from the meta population. Probability of
          each policy is calculated as;

          .. math :: 
              P\\left(\\theta^{m}\\right)=\\frac{N\\left(\\theta^{m}, A\\right)}{\\sum_{j=1}^{M} N\\left(\\theta^{3}, A\\right)}

        k: Number of nearest neigbhours used in the calculation of the novelty.
        device: Torch device

            .. note ::

                For every process a target network will be created to use during rollout.
                That is why I don't recommend use of ``torch.device('cuda')``.
        policy_kwargs: This dictionary of arguments will passed to the policy during 
                       initialization.
        agent_kwargs: This dictionary of arguments will passed to the agent during 
                      initialization.
        optimizer_kwargs: This dictionary of arguments will passed to
                          the optimizer during initialization.
                         
    :var meta_population: List of (policy, optimizer) tuples.
    :var idx: Selected (policy, optimizer) tuple index in the current step.
    :var agent: Used for rollout in each processes.
    :var n_parameters: Number of trainable parameters.
    :var best_reward: Best reward achived during the training.
    :var episode_reward: Reward of the chosen policy after the optimization.
    :var best_policy_dict: PyTorch ``state_dict`` of the policy with the highest reward.
    :var population_returns: List of (novelty, reward) tuple of the current population.
    :var population_parameters: Parameter vectors of the current
         population that sampled from the chosen policy.
    """
    _ALGORITHM_TYPE = _Algorithm.novelty
    def __init__(self, policy, agent, optimizer, population_size, sigma=0.01,
                 meta_population_size=3, k=10, device=torch.device("cpu"),
                 policy_kwargs={}, agent_kwargs={}, optimizer_kwargs={}):

        super().__init__(policy, agent, optimizer, population_size, sigma,
                         device, policy_kwargs, agent_kwargs, optimizer_kwargs)

        self.meta_population_size = meta_population_size
        self.k = k

        if self.rank == 0:
            self._archive = []
            self.meta_population = []
            for _ in range(self.meta_population_size):
                p = policy(**policy_kwargs).to(self.device)
                optim = optimizer(p.parameters(), **optimizer_kwargs)
                self.meta_population.append((p, optim))
                reward, bc = self.agent.rollout(p)
                if bc is None:
                    raise ValueError("Behaviour Charateristics is None")
                self._archive.append(bc)
        else:
            self._archive = None

    def _calculate_novelty(self, bc, _archive):
        kd = spatial.cKDTree(_archive)
        distances, idxs = kd.query(bc, k=self.k)
        distances = distances[distances < float('inf')]
        novelty = np.sum(distances) / np.linalg.norm(_archive)
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
        self._archive.append(bc)
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self.best_policy_dict = policy.state_dict()

    def _calculate_returns(self, parameters):
        returns = []
        for parameter in parameters:
            torch.nn.utils.vector_to_parameters(
                parameter.to(self.device), self.target.parameters())
            reward, bc = self.agent.rollout(self.target)
            novelty = self._calculate_novelty(bc, self._archive)
            returns.append((reward, novelty))
        return np.array(returns, dtype=np.float32)

    def _get_policy(self):
        total_novelty = []
        for policy, _ in self.meta_population:
            reward, bc = self.agent.rollout(policy)
            novelty = self._calculate_novelty(bc, self._archive)
            total_novelty.append(novelty)
        total_novelty = np.array(total_novelty)
        meta_population_probability = total_novelty / np.sum(total_novelty)
        self.idx = np.random.choice(
            np.arange(len(self.meta_population), dtype=np.int),
            p=meta_population_probability)
        policy, optimizer = self.meta_population[self.idx]
        return policy, optimizer

    def _send_to_slaves(self, split_parameters):
        for i in range(1, self.n_workers):
            self._comm.Send(split_parameters[i].numpy(), dest=i)
        self._comm.bcast(self._archive, root=0)

    def _recv_from_master(self):
        parameters = np.empty((int(self.population_size/self.n_workers),
                               self.n_parameters), dtype=np.float32)
        self._comm.Recv(parameters, source=0, status=self.status)
        tag = self.status.Get_tag()
        if tag == _Tag.STOP:
            return
        self._archive = self._comm.bcast(self._archive, root=0)
        parameters = torch.from_numpy(parameters).float()
        return parameters


class NSR_ES(NS_ES):
    """Quality Diversity Evolution Strategy Algorithm. It optimizes
    given policy for the max avarage of novelty and reward return. For
    example usage refer to
    https://github.com/goktug97/estorch/blob/master/examples/nsra_es.py
    This class is inherited from the :class:`NS_ES` which inherits
    from :class:`ES` so every function that is described in the
    :class:`ES` can be used in this class too.

    .. math::

        \\theta_{t+1}^{m} \\leftarrow \\theta_{t}^{m}+\\alpha \\frac{1}{n \\sigma} \\sum_{i=1}^{n} \\frac{f\\left(\\theta_{t}^{i, m}\\right)+N\\left(\\theta_{t}^{i, m}, A\\right)}{2} \\epsilon_{i}    

    - Improving Exploration in Evolution Strategies for Deep
      Reinforcement Learning via a Population of Novelty-Seeking Agents
      http://papers.nips.cc/paper/7750-improving-exploration-in-evolution-strategies-for-deep-reinforcement-learning-via-a-population-of-novelty-seeking-agents.pdf

    Args:
        policy: PyTorch Module. Should be passed as a ``class``.
        agent: Policy will be optimized to maximize the output of this
               class's rollout function. For an example agent class refer to;
               https://github.com/goktug97/estorch/blob/master/examples/cartpole_es.py
               Should be passed as a ``class``.
        optimizer: Optimizer that will be used to update parameters of the policy.
                   Any PyTorch optimizer can be used. Should be passed as a ``class``.
        population_size: Population size of the evolution strategy.

            .. note ::

                if you are using multiprocessing make sure ``population_size`` is
                multiple of ``n_proc``
        sigma: Standart Deviation to use while sampling the generation from the policy.
        meta_population_size: Instead of one policy a meta population
          of policies are optimized during
          training. Each step a policy is chosen
          from the meta population. Probability of
          each policy is calculated as;

          .. math :: 
              P\\left(\\theta^{m}\\right)=\\frac{N\\left(\\theta^{m}, A\\right)}{\\sum_{j=1}^{M} N\\left(\\theta^{3}, A\\right)}

        k: Number of nearest neigbhours used in the calculation of the novelty.
        device: Torch device

            .. note ::

                For every process a target network will be created to use during rollout.
                That is why I don't recommend use of ``torch.device('cuda')``.
        policy_kwargs: This dictionary of arguments will passed to the policy during 
                       initialization.
        agent_kwargs: This dictionary of arguments will passed to the agent during 
                      initialization.
        optimizer_kwargs: This dictionary of arguments will passed to
                          the optimizer during initialization.
                         
    :var meta_population: List of (policy, optimizer) tuples.
    :var idx: Selected (policy, optimizer) tuple index in the current step.
    :var agent: Used for rollout in each processes.
    :var n_parameters: Number of trainable parameters.
    :var best_reward: Best reward achived during the training.
    :var episode_reward: Reward of the chosen policy after the optimization.
    :var best_policy_dict: PyTorch ``state_dict`` of the policy with the highest reward.
    :var population_returns: List of (novelty, reward) tuple of the current population.
    :var population_parameters: Parameter vectors of the current
         population that sampled from the chosen policy.
    """
    _ALGORITHM_TYPE = _Algorithm.novelty
    def _calculate_grad(self, epsilon):
        ranked_rewards = torch.from_numpy(
            rank_transformation(self.population_returns[:, 0])).unsqueeze(0).float()
        ranked_novelties = torch.from_numpy(rank_transformation(
                self.population_returns[:, 1])).unsqueeze(0).float()
        grad = (torch.mm((ranked_novelties+ranked_rewards)/2, epsilon) /
                (self.population_size * self.sigma)).squeeze()
        return grad


class NSRA_ES(NS_ES):
    """Quality Diversity Evolution Strategy Algorithm. It optimizes
    given policy for the max weighted avarage of novelty and reward return. For
    example usage refer to
    https://github.com/goktug97/estorch/blob/master/examples/nsra_es.py
    This class is inherited from the :class:`NS_ES` which inherits
    from :class:`ES` so every function that is described in the
    :class:`ES` can be used in this class too.

    .. math::

        \\theta_{t+1}^{m} \\leftarrow \\theta_{t}^{m}+\\alpha \\frac{1}{n \\sigma} \\sum_{i=1}^{n} w f\\left(\\theta_{t}^{i, m}\\right) \\epsilon_{i}+(1-w) N\\left(\\theta_{t}^{i, m}, A\\right) \\epsilon_{i}

    - Improving Exploration in Evolution Strategies for Deep
      Reinforcement Learning via a Population of Novelty-Seeking Agents
      http://papers.nips.cc/paper/7750-improving-exploration-in-evolution-strategies-for-deep-reinforcement-learning-via-a-population-of-novelty-seeking-agents.pdf

    Args:
        policy: PyTorch Module. Should be passed as a ``class``.
        agent: Policy will be optimized to maximize the output of this
               class's rollout function. For an example agent class refer to;
               https://github.com/goktug97/estorch/blob/master/examples/cartpole_es.py
               Should be passed as a ``class``.
        optimizer: Optimizer that will be used to update parameters of the policy.
                   Any PyTorch optimizer can be used. Should be passed as a ``class``.
        population_size: Population size of the evolution strategy.

            .. note ::

                if you are using multiprocessing make sure ``population_size`` is
                multiple of ``n_proc``
        sigma: Standart Deviation to use while sampling the generation from the policy.
        meta_population_size: Instead of one policy a meta population
          of policies are optimized during
          training. Each step a policy is chosen
          from the meta population. Probability of
          each policy is calculated as;

          .. math :: 
              P\\left(\\theta^{m}\\right)=\\frac{N\\left(\\theta^{m}, A\\right)}{\\sum_{j=1}^{M} N\\left(\\theta^{3}, A\\right)}

        k: Number of nearest neigbhours used in the calculation of the novelty.
        min_weight,weight_t,weight_delta: If the max reward doesn't improve for 
            ``weight_t`` the :attr:`weight` is lowered by
            ``weight_delta`` amount. It can't get lower than
            ``min_weight``.
        device: Torch device

            .. note ::

                For every process a target network will be created to use during rollout.
                That is why I don't recommend use of ``torch.device('cuda')``.
        policy_kwargs: This dictionary of arguments will passed to the policy during 
                       initialization.
        agent_kwargs: This dictionary of arguments will passed to the agent during 
                      initialization.
        optimizer_kwargs: This dictionary of arguments will passed to
                          the optimizer during initialization.
                         
    :var meta_population: List of (policy, optimizer) tuples.
    :var idx: Selected (policy, optimizer) tuple index in the current step.
    :var agent: Used for rollout in each processes.
    :var n_parameters: Number of trainable parameters.
    :var best_reward: Best reward achived during the training.
    :var episode_reward: Reward of the chosen policy after the optimization.
    :var best_policy_dict: PyTorch ``state_dict`` of the policy with the highest reward.
    :var population_returns: List of (novelty, reward) tuple of the current population.
    :var population_parameters: Parameter vectors of the current
         population that sampled from the chosen policy.
    """
    _ALGORITHM_TYPE = _Algorithm.novelty
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
        self._archive.append(bc)
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self.weight = min(self.weight + self.weight_delta, 1.0)
            self.best_policy_dict = policy.state_dict()
            self.t = 0
        else:
            self.t += 1
            if self.t >= self.weight_t:
                self.weight = max(self.weight - self.weight_delta, self.min_weight)
                self.t = 0
