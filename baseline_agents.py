import numpy as np
from ucb_agents import *

### Stationary AR Bandit (Bacchiocchi et. al, 2022) ###
class StatAR(UCBAgent):
    def __init__(self, state_dim, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__("Stat. AR", state_dim, alpha, lambda_reg, num_actions)
        self.k = int((state_dim - 1) / (2 * num_actions))

    def process_states(self, env, actions, rewards):
        K = self.k
        t = env.get_t()
        recent_rewards = rewards[t - K:t]

        return np.insert(np.copy(recent_rewards), 0, 1)

### Sliding Window UCB (Garivier & Moulines, 2008) ###
# uses a sliding window approach of size tau = k
# code adapted from: https://github.com/MaxenceGiraud/ucb-nonstationary/blob/main/nsucb/sliding_ucb.py
class SW_UCB(UCBAgent):
    def __init__(self, k, num_actions=2):
        super().__init__("SW-UCB", 1, ALPHA, LAMBDA_REG, num_actions)
        self.gamma = 0.9
        self.tau = k
        self.B = 100 # upper bound on rewards
        self.xi = 0.5 # original authors set 0.5 for simulations
        self.optimism_values = np.zeros(self.num_actions)

    def process_states(self, env, actions, rewards):
        return np.ones(1)
    
    def select_action(self, action_set):
        return np.argmax([optimism_a for optimism_a in self.optimism_values])
    
    def update(self, actions, rewards, action_sets, t):
        for i in range(self.num_actions):
            N = np.sum(actions[-self.tau:])
            X = (1 / N) * np.sum(rewards[-self.tau:])
            c = self.B * np.sqrt((self.xi * np.log(max(self.t, self.tau))) / N)
            self.optimism_values[i] = X + c

### Predictive Sampling (Liu et al., 2023) ###
class PredictiveSampling:
    def __init__(self, num_actions=2):
        self.name = "PS"
        self.num_actions = num_actions
        self.mu_squiggle = np.zeros(self.num_actions)
        self.Sigma_squiggle = np.eye(self.num_actions)

    def process_states(self, env, actions, rewards):
        return np.ones(1)
    
    def select_action(self, action_set):
        sampled_theta = np.random.multivariate_normal(self.mu_squiggle, self.Sigma_squiggle)
        return np.argmax([theta_a for theta_a in sampled_theta])
    
    def update(self, actions, rewards, action_sets, t):
        action = actions[t]
        reward = rewards[t]        
        ## ANNA TODO