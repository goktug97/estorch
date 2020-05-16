import torch
import numpy as np
import estorch

from cartpole_es import Policy, Agent

class ES(estorch.ES):
    def log(self):
        """This function is executed after every optimization. So it can be used 
        to interact with the system during training."""
        # For the purpose of this example, we will stop the training if any of
        # the policies from the current generations' population gets the maximum
        # reward it can get from environment. For this example we will use CartPole-v1,
        # where you can get maximum of 500 rewards.

        # For the ES the population_returns is rewards
        # For NS Algorithms it is a list of (reward, novelty) tuples.
        idx = np.argmax(self.population_returns)
        reward = self.population_returns[idx, 0]
        print(f'Reward: {reward}')
        if reward == 500:
            self.best = self.population_parameters[idx]
            self.terminate()

if __name__ == '__main__':
    device = torch.device("cpu")
    agent = Agent()
    n_input = agent.env.observation_space.shape[0]
    n_output = agent.env.action_space.n
    es = ES(Policy, Agent, torch.optim.Adam, population_size=100, sigma=0.02,
            device=device, policy_kwargs={'n_input': n_input, 'n_output': n_output},
            agent_kwargs={'device': device}, optimizer_kwargs={'lr': 0.01})
    es.train(n_steps=1000, n_proc=2)

    policy = Policy(n_input, n_output).to(device)
    torch.nn.utils.vector_to_parameters(
        es.best.to(device), policy.parameters())
    reward = agent.rollout(policy, render=True)
    print(f'\nReward: {reward}')
