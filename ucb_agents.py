import numpy as np

### Helper Functions ###
def context_to_action_states(state, num_actions):
    state_length = len(state)
    result = np.zeros((num_actions, num_actions * state_length), dtype=int)
    for a in range(num_actions):
        start_index = state_length * a
        end_index = start_index + state_length
        result[a, start_index:end_index] = state

    return result

# def stack_thetas(theta_hats):
#     theta = np.vstack(theta_hats)

#     return theta

### Parent Agent Class ###
class UCBAgent:
    def __init__(self, name, state_dim, lambda_reg=1.0, num_actions=2):
        self.name = name
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.lambda_reg = lambda_reg
        self.V = self.lambda_reg * np.eye(self.state_dim * num_actions)
        self.b = np.zeros((self.state_dim * num_actions, 1))
        self.theta_hat = np.linalg.inv(self.V) @ self.b

    def process_state(self, env, rewards):
        return None

    # upper confidence provided by thie article: https://www.linkedin.com/pulse/contextual-bandits-linear-upper-confidence-bound-disjoint-kenneth-foo/
    def select_action(self, state):
        action_set = context_to_action_states(state, self.num_actions)
        # Compute the optimistic estimates for each arm
        optimism = [self.lambda_reg * np.sqrt(np.dot(action_state.T, np.dot(np.linalg.inv(self.V), action_state)))
                     for action_state in action_set]
        optimistic_estimates = [np.dot(self.theta_hat.T, action_state) + optimism_value
                                for action_state, optimism_value in zip(action_set, optimism)]
        print("MEANS!!", [np.dot(self.theta_hat.T, action_state) for action_state in action_set])
        print("CONFIDENCE!!!", optimism)
        print("OPTIMISTIC ESTIMATES!", optimistic_estimates)

        # Choose the arm with the maximum optimistic estimate
        chosen_action = np.argmax(optimistic_estimates)
        return chosen_action
    
    def update(self, action, reward, state):
        # Update V and b for the chosen arm using Bayesian linear regression
        action_set = context_to_action_states(state, self.num_actions)
        feature = action_set[int(action)].reshape((-1, 1))
        self.V += np.dot(feature, feature.T)
        self.b += reward * feature
        self.theta_hat = np.linalg.inv(self.V) @ self.b
        print("THETA HAT!", self.theta_hat)
    
    def get_state_dim(self):
        return self.state_dim
    
### Helpers ###
    
### Standard (Stationary) MAB ###
class StationaryAgent(UCBAgent):
    def __init__(self, num_actions=2):
        super().__init__("stationary_agent", 1, lambda_reg=10.0, num_actions=num_actions)

    def process_state(self, env, rewards):
        return 1

### Naive UCB for Non-Stat Latent Autoregressive Env. ###
class NaiveNSLAR(UCBAgent):
    def __init__(self, state_dim, lambda_reg=1.0, num_actions=2):
        super().__init__("naive_non_stat", state_dim, lambda_reg, num_actions)

    def process_state(self, env, rewards):
        K = env.K
        t = env.get_t()
        return np.insert(rewards[t - K:t], 0, 1)
    
### Weighted UCB for Non-Stat Latent Autoregressive Env. ###
class WeightedNSLAR(UCBAgent):
    def __init__(self, state_dim, lambda_reg=1.0, num_actions=2):
        super().__init__("naive_non_stat", state_dim, lambda_reg, num_actions)

    def process_state(self, env, rewards):
        K = env.K
        t = env.get_t()
        return np.insert(rewards[t - K:t], 0, 1)
    
### Oracle that knows the ground-truth parameters but does not observe the latent process ###
class NonStatOracle(UCBAgent):
    def __init__(self, env_params, state_dim, lambda_reg=1.0, num_actions=2):
        super().__init__("non_stat_oracle", state_dim, lambda_reg, num_actions)
        gammas = np.array(env_params["gammas"])
        gamma_0 = env_params["gamma_0"]
        beta_0 = env_params["beta_0"]
        beta_1 = env_params["beta_1"]
        beta_tilde = lambda a: beta_0[a] + beta_1[a] * gamma_0 - beta_0[a] * np.sum(gammas)
        self.theta_hats = [np.concatenate((gammas, beta_tilde(a)), axis=None) for a in range(num_actions)]

    def process_state(self, env, rewards):
        K = env.K
        t = env.get_t()
        return np.insert(rewards[t - K:t], 0, 1)
    
    def select_action(self, state):
        mean_rewards = [state @ theta for theta in self.theta_hats]
        chosen_arm = np.argmax(mean_rewards)
        return chosen_arm
    
    # oracle is given the ground-truth parameters and do not need to update
    def update(self, action, reward, state):
        return None