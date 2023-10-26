import random
import numpy as np

NUM_SAMPLES = 500

### Parent Agent Class ###
class Agent:
    def __init__(self, name, prior_mean, prior_var, num_actions=2):
        self.name = name
        self.num_actions = num_actions
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.state_dim = 1 if np.isscalar(prior_mean) else len(prior_mean)

    def process_state(self, env, rewards):
        return None

    def select_action(self, state):
        """
        Select an action based on the current state. You can customize this method
        for your specific problem and policy.
        """
        return None
    
    def update(self):
        return None
    
    def get_state_dim(self):
        return self.state_dim
    
### Helpers ###
def compute_posterior_var(Phi, noise_var, prior_var):
  return np.linalg.inv(1/noise_var * Phi.T @ Phi + np.linalg.inv(prior_var))

def compute_posterior_mean(Phi, R, noise_var, prior_mean, prior_var, posterior_var):
  return posterior_var @ (1/noise_var * Phi.T @ R + np.linalg.inv(prior_var) @ prior_mean)
    
### Standard (Stationary) Thompson Sampler without context ###
class StationaryAgent(Agent):
    def __init__(self, prior_mean, prior_var, num_actions=2):
        super().__init__("stationary_agent", prior_mean, prior_var, num_actions)
        self.posterior_means = [prior_mean] * num_actions
        self.posterior_vars = [prior_var] * num_actions

    # does not use state to perform action selection
    def select_action(self, state):
        # note: np.random.normal takes in standard deviation and not variance: 
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        sampled_values = [np.mean(np.random.normal(self.posterior_means[a], self.posterior_vars[a]**0.5, NUM_SAMPLES)) for a in range(self.num_actions)]
        return sampled_values.index(max(sampled_values))

    # does not use states in update
    def update(self, actions, rewards, states, noise_var):
        for a in range(self.num_actions):
            R = rewards[np.where(actions == a)]
            if len(R) > 0:
                self.posterior_vars[a] = ((1 / self.prior_var) + (len(R) / noise_var))**(-1)
                self.posterior_means[a] = self.posterior_vars[a] * ((self.prior_mean / self.prior_var) + (np.mean(R) / noise_var))

### Naive Thompson Sampler for Non-Stat Latent Autoregressive Env. ###
class NaiveNlaTS(Agent):
    def __init__(self, prior_mean, prior_var, num_actions=2):
        super().__init__("naive_non_stat", prior_mean, prior_var, num_actions)
        self.posterior_means = [prior_mean] * num_actions
        self.posterior_vars = [prior_var] * num_actions

    def process_state(self, env, rewards):
        K = env.K
        t = env.get_t()
        return np.insert(rewards[t - K:t], 0, 1)

    def reward_approx_func(self, weights, state):
        return weights @ state

    # state is the most recent K rewards
    def select_action(self, state):
        sampled_values = [np.random.multivariate_normal(self.posterior_means[a], self.posterior_vars[a], NUM_SAMPLES) for a in range(self.num_actions)]
        estimated_rewards = [np.mean([self.reward_approx_func(sampled_values[a][i], state) for i in range(NUM_SAMPLES)]) for a in range(self.num_actions)]

        return np.argmax(estimated_rewards)
    
    # does not use latent states in update
    def update(self, actions, rewards, states, noise_var):
        for a in range(self.num_actions):
            Phi = states[np.where(actions == a)]
            R = rewards[np.where(actions == a)]
            self.posterior_vars[a] = compute_posterior_var(Phi, noise_var, self.prior_var)
            self.posterior_means[a] = compute_posterior_mean(Phi, R, noise_var, self.prior_mean, self.prior_var, self.posterior_vars[a])
    
### Oracle that observes the latent process and uses it as its state ###
class Oracle(Agent):
    def __init__(self, prior_mean, prior_var, num_actions=2):
        super().__init__("oracle", prior_mean, prior_var, num_actions)
        self.posterior_means = [prior_mean] * num_actions
        self.posterior_vars = [prior_var] * num_actions

    def process_state(self, env, rewards):
        return np.array([1, env.get_all_zs()[env.get_t()]])

    def reward_approx_func(self, weights, state):
        return weights @ state

    # state is z_t
    def select_action(self, state):
        sampled_values = [np.random.multivariate_normal(self.posterior_means[a], self.posterior_vars[a], NUM_SAMPLES) for a in range(self.num_actions)]
        estimated_rewards = [np.mean([self.reward_approx_func(sampled_values[a][i], state) for i in range(NUM_SAMPLES)]) for a in range(self.num_actions)]
        
        return np.argmax(estimated_rewards)
    
    def update(self, actions, rewards, states, noise_var):
        for a in range(self.num_actions):
            Phi = states[np.where(actions == a)]
            R = rewards[np.where(actions == a)]
            self.posterior_vars[a] = compute_posterior_var(Phi, noise_var, self.prior_var)
            self.posterior_means[a] = compute_posterior_mean(Phi, R, noise_var, self.prior_mean, self.prior_var, self.posterior_vars[a])