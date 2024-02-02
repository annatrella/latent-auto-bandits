import numpy as np

class Environment:
    def __init__(self, params, T):
        self.K = params["K"]
        self.noise_std = params["noise_std"] # std on the latent state
        self.gamma_0 = params["gamma_0"]
        self.gammas = np.array(params["gammas"]) # gammas in the sum, size: (K, 1)
        self.beta_0 = params["beta_0"] # size: (num_actions, 1)
        self.beta_1 = params["beta_1"] # size: (num_actions, 1)
        self.init_zs = params["init_zs"]
        self.num_actions = len(self.beta_0)
        self.T = T # num. time-steps
        self.t = 0 # current time-step
        self.zs = np.concatenate((self.init_zs, np.empty(T - self.K, dtype=np.float64)), axis=None)
        self.z_noises = np.empty(T)
        self.reward_noises = np.empty(T, dtype=np.float64)

    def get_num_actions(self):
        return self.num_actions

    def increment_t(self):
        self.t += 1

    def state_evolution(self):
        # order goes from t - k to t - 1
        dot_prod = self.gammas @ np.array(self.get_recent_zs())
        z_noise = np.random.normal(0, self.noise_std)
        t = self.get_t()
        self.z_noises[t] = z_noise

        self.zs[self.t] = self.gamma_0 + dot_prod + z_noise

    def get_reward(self, action):
        t = self.get_t()
        z_t = self.zs[t]
        reward_noise = np.random.normal(0, 0.1)
        self.reward_noises[t] = reward_noise
        return self.beta_0[action] + self.beta_1[action] * z_t + reward_noise
    
    # for sanity checking performance of oracle
    def get_noiseless_reward(self, action, t):
        z_t = self.zs[t]
        return self.beta_0[action] + self.beta_1[action] * z_t
    
    def get_reward_noise(self, t):
        return self.reward_noises[t]
    
    def get_all_zs(self):
        return self.zs

    # returns the most recent K Zs
    # [z_{t - k}, ..., z_{t - 1}]
    def get_recent_zs(self):
        t = self.get_t()
        return self.zs[t - self.K:t]
    
    def get_t(self):
        return self.t 
    
    def get_T(self):
        return self.T
    
    def get_noise_std(self):
        return self.noise_std