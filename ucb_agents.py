import numpy as np

### Parent Agent Class ###
class UCBAgent:
    def __init__(self, name, state_dim, lambda_reg=1.0, num_actions=2):
        self.name = name
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.lambda_reg = lambda_reg
        self.V_invs = [self.lambda_reg * np.eye(self.state_dim) for _ in range(self.num_actions)]
        self.bs = [np.zeros((self.state_dim, 1)) for _ in range(self.num_actions)]
        self.theta_hats = [np.linalg.inv(V_inv) @ b for V_inv, b in zip(self.V_invs, self.bs)]

    def process_state(self, env, rewards):
        return None

    def select_action(self, state):
        # Compute the optimistic estimates for each arm
        optimism = [np.sqrt(np.dot(state.T, np.dot(np.linalg.inv(V_inv), state)))
                     for V_inv in self.V_invs]
        optimistic_estimates = [np.dot(theta.T, state) + optimism_value
                                for theta, optimism_value in zip(self.theta_hats, optimism)]

        # Choose the arm with the maximum optimistic estimate
        chosen_arm = np.argmax(optimistic_estimates)
        return chosen_arm
    
    def update(self, action, reward, state):
        # Update A_inv and b for the chosen arm using Bayesian linear regression
        action = int(action)
        feature = state.reshape((-1, 1))
        self.V_invs[action] += np.dot(feature, feature.T)
        self.bs[action] += reward * feature
        self.theta_hats = [np.linalg.inv(V_inv) @ b for V_inv, b in zip(self.V_invs, self.bs)]
    
    def get_state_dim(self):
        return self.state_dim
    
### Helpers ###
    
### Standard (Stationary) MAB ###
class StationaryAgent(UCBAgent):
    def __init__(self, num_actions=2):
        super().__init__("stationary_agent", 1, lambda_reg=1.0, num_actions=num_actions)

    def process_state(self, env, rewards):
        return 1

### Naive UCB for Non-Stat Latent Autoregressive Env. ###
class NaiveNlaTS(UCBAgent):
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