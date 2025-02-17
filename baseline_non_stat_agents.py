import numpy as np
from ucb_agents import *

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