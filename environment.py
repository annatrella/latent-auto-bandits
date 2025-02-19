import numpy as np

class Environment:
    def __init__(self, params, T):
        self.K = params["K"]
        self.sigma_z = params["sigma_z"] # std on the latent state
        self.sigma_r = params["sigma_r"]
        self.gamma_0 = params["gamma_0"]
        self.gammas = np.array(params["gammas"]) # gammas in the sum, size: (K, 1)
        self.mu_a = params["mu_a"] # size: (num_actions, 1)
        self.beta_a = params["beta_a"] # size: (num_actions, 1)
        self.init_zs = params["init_zs"]
        self.num_actions = len(self.mu_a)
        self.T = T # num. time-steps
        self.t = 0 # current time-step
        self.zs = np.concatenate((self.init_zs, np.zeros(T - self.K, dtype=np.float64)), axis=None)
        self.z_noises = np.zeros(T)
        self.reward_noises = np.zeros(T, dtype=np.float64)

    def get_num_actions(self):
        return self.num_actions

    def increment_t(self):
        self.t += 1

    def state_evolution(self):
        # order goes from t - 1 to t - k / from gamma_1 to gamma_k
        dot_prod = self.gammas @ np.flip(np.array(self.get_recent_zs()))
        z_noise = np.random.normal(0, self.sigma_z)
        t = self.get_t()
        self.z_noises[t] = z_noise

        self.zs[self.t] = self.gamma_0 + dot_prod + z_noise

    def get_reward(self, action):
        t = self.get_t()
        z_t = self.zs[t]
        reward_noise = np.random.normal(0, self.sigma_r)
        self.reward_noises[t] = reward_noise
        return self.mu_a[action] + self.beta_a[action] * z_t + reward_noise
    
    # for sanity checking performance of oracle
    def get_noiseless_reward(self, action, t):
        z_t = self.zs[t]
        return self.mu_a[action] + self.beta_a[action] * z_t
    
    def get_reward_noise(self, t):
        return self.reward_noises[t]
    
    def get_all_zs(self):
        return self.zs

    # returns the most recent K Zs
    # [z_{t - k}, ..., z_{t - 1}]
    def get_recent_zs(self):
        t = self.get_t()
        return self.zs[t - self.K:t]
    
    def get_k(self):
        return self.K
    
    def get_t(self):
        return self.t 
    
    def get_T(self):
        return self.T
    
    def get_sigma_z(self):
        return self.sigma_z