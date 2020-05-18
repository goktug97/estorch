import gym
import torch
from estorch import ES, rank_transformation

from cartpole_es import Policy, Agent


class SymmetricES(ES):
    """Custom Evolution Strategy algorithm that samples parameters symmetrically and
    updates parameters similar to the parameter exploring policy gradients algorithm."""

    def _sample_policy(self, policy):
        parameters = torch.nn.utils.parameters_to_vector(policy.parameters())
        normal = torch.distributions.normal.Normal(0, self.sigma)
        epsilon = normal.sample([int(self.population_size/2), parameters.shape[0]])
        parameters = parameters.detach().cpu()
        population_parameters = torch.cat((parameters + epsilon, parameters - epsilon))
        return population_parameters, epsilon

    def _calculate_grad(self, epsilon):
        ranked_rewards = torch.from_numpy(
            rank_transformation(self.population_returns.squeeze())).unsqueeze(0).float()
        batch = int(self.population_size/2)
        grad = (torch.mm(
            (ranked_rewards[0, :batch]-ranked_rewards[0, batch:]).unsqueeze(0), epsilon) /
                ((self.population_size/2) * self.sigma)).squeeze()
        return grad

if __name__ == '__main__':
    device = torch.device("cpu")
    agent = Agent()
    n_input = agent.env.observation_space.shape[0]
    n_output = agent.env.action_space.n
    es = SymmetricES(Policy, Agent, torch.optim.Adam,
                     population_size=100, sigma=0.02, device=device,
                     policy_kwargs={'n_input': n_input, 'n_output': n_output},
                     agent_kwargs={'device': device}, optimizer_kwargs={'lr': 0.01})
    es.train(n_steps=100, n_proc=2)

    # Policy with the highest reward
    policy = Policy(n_input, n_output).to(device)
    policy.load_state_dict(es.best_policy_dict)
    reward = agent.rollout(policy, render=True)
    print(f'Best Policy Reward: {reward}')

