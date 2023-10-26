import numpy as np

class Environment:
    def __init__(self, params, T):
        self.K = params["K"]
        self.noise_var = params["noise_var"] # std
        self.gamma_0 = params["gamma_0"]
        self.gammas = np.array(params["gammas"]) # gammas in the sum, size: (K, 1)
        self.beta_0 = params["beta_0"] # size: (num_actions, 1)
        self.beta_1 = params["beta_1"] # size: (num_actions, 1)
        self.init_zs = params["init_zs"]
        self.T = T # num. time-steps
        self.t = self.K # current time-step, starts with K
        self.zs = np.concatenate((self.init_zs,np.empty(T - self.K)), axis=None)

    def increment_t(self):
        self.t += 1

    def state_evolution(self):
        dot_prod = self.gammas @ np.array(self.get_recent_zs())

        self.zs[self.t] = self.gamma_0 + dot_prod + np.random.normal(0, self.noise_var)

    def get_reward(self, action):
        z_t = self.zs[self.t]
        return self.beta_0[action] + self.beta_1[action] * z_t + np.random.normal(0, self.noise_var)
    
    # for sanity checking performance of oracle
    def get_noiseless_reward(self, action):
        z_t = self.zs[self.t]
        return self.beta_0[action] + self.beta_1[action] * z_t
    
    def get_all_zs(self):
        return self.zs

    # returns the most recent K Zs
    def get_recent_zs(self):
        t = self.get_t()
        return self.zs[t - self.K:t]
    
    def get_t(self):
        return self.t 
    
    def get_T(self):
        return self.T
    
    def get_noise_var(self):
        return self.noise_var