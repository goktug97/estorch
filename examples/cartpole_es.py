import gym
import torch
from estorch import ES

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

class Agent():
    def __init__(self, device=torch.device('cpu')):
        self.env = gym.make('CartPole-v1')
        self.device = device

    def rollout(self, policy, render=False):
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
        return total_reward

if __name__ == '__main__':
    device = torch.device("cpu")
    agent = Agent()
    n_input = agent.env.observation_space.shape[0]
    n_output = agent.env.action_space.n
    es = ES(Policy, Agent, torch.optim.Adam, population_size=100, sigma=0.02,
            device=device, policy_kwargs={'n_input': n_input, 'n_output': n_output},
            agent_kwargs={'device': device}, optimizer_kwargs={'lr': 0.01})
    es.train(n_steps=100, n_proc=2)

    # Latest Policy
    reward = agent.rollout(es.policy, render=True)
    print(f'Latest Policy Reward: {reward}')

    # Policy with the highest reward
    policy = Policy(n_input, n_output).to(device)
    policy.load_state_dict(es.best_policy_dict)
    reward = agent.rollout(policy, render=True)
    print(f'Best Policy Reward: {reward}')
