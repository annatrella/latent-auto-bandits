import numpy as np

# Sanity check stationary environment with 2 arms 
class StationaryEnvironment:
    def __init__(self, T=100, d=12, sigma_z=0.1, seed=42):
        np.random.seed(seed)
        self.T = T
        self.t = 0
        self.d = d
        self.sigma_z = sigma_z
        self.X = np.random.rand(T, d)
        self.num_actions = 2
        self.true_theta = np.random.randn(self.num_actions * d)
        self.true_reward_means = np.vstack((self.X @ self.true_theta[:d], self.X @ self.true_theta[d:]))
        assert self.true_reward_means.shape == (self.num_actions, T)
        self.noisy_rewards = self.true_reward_means + np.random.normal(0, sigma_z, T)
        assert self.noisy_rewards.shape == (self.num_actions, T)

    def get_num_actions(self):
        return self.num_actions

    def increment_t(self):
        self.t += 1

    def get_state(self):
        t = self.get_t()
        return self.X[t]

    def get_reward(self, action):
        t = self.get_t()
        return self.noisy_rewards[action][t]
    
    def get_noiseless_reward(self, action):
        t = self.get_t()
        return self.true_reward_means[action][t]
    
    def get_t(self):
        return self.t
    
    def get_T(self):
        return self.T
    
    def sigma_z(self):
        return self.sigma_z