import numpy as np
from ucb_agents import *

### Rexp3 (Besbes et al., 2014) ###
# paper: https://proceedings.neurips.cc/paper_files/paper/2014/file/903ce9225fca3e988c2af215d4e544d3-Paper.pdf
class Rexp3:
    def __init__(self, T, V_t, num_actions=2):
        self.name = "Rexp3"
        self.num_actions = num_actions
        self.state_dim = 1
        self.weights = np.ones(self.num_actions)
        self.probs = np.ones(self.num_actions) / self.num_actions
        # batch size and gamma are set to the values described in Theorem 2 of paper
        self.batch_size = np.ceil((self.num_actions * np.log(self.num_actions))**(1/3) * (T / V_t)**(2/3))
        print(f"Rexp3 batch size: {self.batch_size}")
        self.gamma = min(1, np.sqrt(self.num_actions * np.log(self.num_actions) / (np.exp(1) * self.batch_size)))

    def get_state_dim(self):
        return self.state_dim

    def process_state(self, env, actions, rewards):
        return None
    
    def select_action(self, env, state):
        return np.random.choice(range(self.num_actions), p=self.probs)
    
    def update_weights(self, action, reward):
        weighted_reward = reward / self.probs[action]
        self.weights[action] *= np.exp(self.gamma * weighted_reward / self.num_actions)

    def update_probs(self):
        self.probs = (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.num_actions

    def update(self, actions, rewards, states, t):
        action = int(actions[t])
        reward = rewards[t]
        # update weights
        self.update_weights(action, reward)
        # update probabilities
        self.update_probs()
        # check if need to restart
        if t % self.batch_size == 0:
            self.weights = np.ones(self.num_actions)
            self.probs = np.ones(self.num_actions) / self.num_actions

### Predictive Sampling (Liu et al., 2023) ###
class PredictiveSampling:
    def __init__(self, num_actions=2):
        self.name = "PS"
        self.num_actions = num_actions
        self.mu_squiggle = np.zeros(self.num_actions)
        self.Sigma_squiggle = np.eye(self.num_actions)

    def process_states(self, env, actions, rewards):
        return np.ones(1)
    
    def select_action(self, env, state):
        sampled_theta = np.random.multivariate_normal(self.mu_squiggle, self.Sigma_squiggle)
        return np.argmax([theta_a for theta_a in sampled_theta])
    
    def update(self, actions, rewards, states, t):
        action = actions[t]
        reward = rewards[t]        
        ## ANNA TODO