import numpy as np
from sklearn.linear_model import Lasso
from rls import fit_rls

LAMBDA_REG = 1.0
ALPHA = 1.0

### Helper Functions ###
def context_to_action_states(states, num_actions):
    state_length = len(states[0])
    result = np.zeros((num_actions, num_actions * state_length), dtype=np.float64)
    for a in range(num_actions):
        start_index = state_length * a
        end_index = start_index + state_length
        result[a, start_index:end_index] = states[a]

    return result

### Parent Agent Class ###
class UCBAgent:
    def __init__(self, name, state_dim, alpha, lambda_reg, num_actions=2):
        self.name = name
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.alpha = alpha # exploration parameter from LinUCB
        self.lambda_reg = lambda_reg
        self.V = self.lambda_reg * np.eye(self.state_dim * num_actions, dtype=np.float64)
        self.b = np.zeros(self.state_dim * num_actions, dtype=np.float64)
        self.theta_hat = np.linalg.inv(self.V) @ self.b

    def process_states(self, env, actions, rewards):
        return None

    # upper confidence provided by thie article: https://www.linkedin.com/pulse/contextual-bandits-linear-upper-confidence-bound-disjoint-kenneth-foo/
    def select_action(self, action_set):
        # Compute the optimistic estimates for each arm
        optimism = [self.alpha * np.sqrt(np.dot(action_state.T, np.dot(np.linalg.inv(self.V), action_state)))
                     for action_state in action_set]
        optimistic_estimates = [np.dot(self.theta_hat.T, action_state) + optimism_value
                                for action_state, optimism_value in zip(action_set, optimism)]

        # Choose the arm with the maximum optimistic estimate
        chosen_action = np.argmax(optimistic_estimates)
        return chosen_action
    
    def update(self, actions, rewards, action_sets, t):
        action = actions[t]
        reward = rewards[t]
        action_set = action_sets[t]
        # Update V and b for the chosen arm using Bayesian linear regression
        feature = action_set[int(action)]
        self.V, self.b, self.theta_hat = fit_rls(self.V, self.b, feature, reward)
    
    def get_state_dim(self):
        return self.state_dim
    
    def get_theta_hat(self):
        return self.theta_hat
    
    def get_V(self):
        return self.V
    
    def set_theta_hat(self, theta_hat):
        assert theta_hat.shape == self.theta_hat
        self.theta_hat = theta_hat
    
### Helpers ###
    
### Standard (Stationary) MAB ###
class StationaryAgent(UCBAgent):
    def __init__(self, k, num_actions=2):
        super().__init__("stationary", 1, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=num_actions)
        self.k = k

    def process_states(self, env, actions, rewards):
        states = np.array(self.num_actions * [1]).reshape(-1, 1)
        action_set = context_to_action_states(states, self.num_actions)
        return action_set
    
### JUST FOR TESTING ###
class StandardRLS(UCBAgent):
    def __init__(self, state_dim, num_actions=2):
        super().__init__("standard_rls_agent", state_dim, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=num_actions)
    
    def process_states(self, env, actions, rewards):
        state = env.get_state()
        states = [state, state]
        action_set = context_to_action_states(states, self.num_actions)

        return action_set
    
class NonStatRLS(UCBAgent):
    def __init__(self, state_dim, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__("our algorithm", state_dim, alpha, lambda_reg, num_actions)
        self.k = int((state_dim - 1) / (2 * num_actions))
        # order goes from gamma_k to gamma_1
        self.bases = np.eye(num_actions).tolist()
        self.reward_means = [0] * self.k

    def process_states(self, env, actions, rewards):
        K = self.k
        t = env.get_t()
        recent_actions = actions[t - K:t]
        recent_rewards = rewards[t - K:t]
        predicted_mean_rewards = np.array(self.reward_means[-K:])
        predicted_noises = recent_rewards - predicted_mean_rewards
        predicted_noises[abs(predicted_noises) > 1] = 1 * np.sign(predicted_noises[abs(predicted_noises) > 1])
        reward_diffs = recent_rewards - predicted_noises
        action_states = np.empty((self.num_actions, 2 * self.num_actions * K + 1))
        for a in range(self.num_actions):
            context = np.concatenate([np.concatenate((reward_diff * np.array(self.bases[int(recent_a)]), -1.0 * np.array(self.bases[int(recent_a)])), axis=None) for recent_a, reward_diff in zip(recent_actions, reward_diffs)], axis=None)
            action_states[a] = np.insert(context, 0, 1)
        assert np.array_equal(action_states[0], action_states[1])
        # print("action states!", action_states[0])
        action_set = context_to_action_states(action_states, self.num_actions)
        return action_set

    def select_action(self, states):
        mean_rewards = [state.T @ self.theta_hat for state in states]
        chosen_arm = np.argmax(mean_rewards)
        self.reward_means.append(mean_rewards[chosen_arm])

        return chosen_arm
    
# our agent but with no corrupted states
class NonStatNoCorruption(UCBAgent):
    def __init__(self, env_params, state_dim, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__("non_stat_no_corruptions", state_dim, alpha, lambda_reg, num_actions)
        self.k = int((state_dim - 1) / (2 * num_actions))
        # order goes from gamma_k to gamma_1
        self.bases = np.eye(num_actions).tolist()
        self.gammas = np.array(env_params["gammas"])
        # gamma_0 = env_params["gamma_0"]
        self.beta_0 = env_params["beta_0"]
        self.beta_1 = env_params["beta_1"]
        # self.beta_tilde = lambda a: self.beta_0[a] + self.beta_1[a] * gamma_0
        # self.reward_and_kappa = lambda j: self.gammas[j] * np.concatenate((1 / self.beta_1[0], 1 / self.beta_1[1], self.beta_0[0] / self.beta_1[0], self.beta_0[1] / self.beta_1[1]), axis=None)
        # self.theta_hat = np.hstack([np.concatenate((self.beta_tilde(a), self.beta_1[int(a)] * np.array([self.reward_and_kappa(j) for j in range(self.k)])), axis=None) for a in range(num_actions)])
        # assert self.theta_hat.shape == (num_actions * (2 * self.k * num_actions + 1), )
        # print("true theta!", self.theta_hat)

    def process_states(self, env, actions, rewards):
        K = self.k
        t = env.get_t()
        recent_actions = actions[t - K:t]
        recent_rewards = rewards[t - K:t]
        recent_reward_diffs = recent_rewards - np.array([env.get_reward_noise(t - (K - j)) for j in range(K)])
        should_be_prev_z = [(recent_reward_diffs[i] - self.beta_0[int(action)]) / self.beta_1[int(action)] for action, i in zip(recent_actions, range(len(recent_actions)))]
        assert np.allclose(should_be_prev_z, env.get_all_zs()[t - K:t])
        action_states = np.empty((self.num_actions, 2 * self.num_actions * K + 1))
        for a in range(self.num_actions):
            context = np.concatenate([np.concatenate((reward_diff * np.array(self.bases[int(recent_a)]), -1.0 * np.array(self.bases[int(recent_a)])), axis=None) for recent_a, reward_diff in zip(recent_actions, recent_reward_diffs)], axis=None)
            action_states[a] = np.insert(context, 0, 1)
        assert np.array_equal(action_states[0], action_states[1])
        action_set = context_to_action_states(action_states, self.num_actions)
        # print("TRUE MEAN REWARD!", (env.get_noiseless_reward(0, t), env.get_noiseless_reward(1, t)))
        return action_set

    def select_action(self, states):
        mean_rewards = [state.T @ self.theta_hat for state in states]
        chosen_arm = np.argmax(mean_rewards)
        return chosen_arm
    
# This agent doesn't know the ground-truth k. It explores for T1 steps and then commits 
# to the learned k.
class ESTCNonStatRLS(NonStatRLS):
    def __init__(self, k_0, T1, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__(2 * 2 * k_0 + 1, alpha, lambda_reg, num_actions)
        self.T1 = T1
        self.k_0 = k_0

    # same as regular linear bandit update, except at T1
    def update(self, actions, rewards, action_sets, t):
        action = actions[t]
        reward = rewards[t]
        action_set = action_sets[t]
        feature = action_set[int(action)]
        self.V, self.b, self.theta_hat = fit_rls(self.V, self.b, feature, reward)
        #### commit to the sparsity ###
        if t == self.T1:
            features = np.array([action_set[int(action)] for action_set, action in zip(action_sets[self.k:t], actions[self.k:t])])
            lasso_model = Lasso(alpha=0.00001)
            lasso_model.fit(features, rewards[self.k:t])
            estimated_coefficients = lasso_model.coef_
            threshold = 1e-3
            print("OLD THETAS!", self.theta_hat)
            print("LASSO THETAS!", estimated_coefficients)
            estimated_coefficients[abs(estimated_coefficients) < threshold] = 0
            print("IT'S TIME!", estimated_coefficients)
            # find \hat{k}
            # set new theta

# only for testing, this agent has access to true parameters but not the mean of the latent process
class ResidualPred(UCBAgent):
    def __init__(self, env_params, state_dim, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__("residual_pred", state_dim, alpha, lambda_reg, num_actions)
        self.k = int((state_dim - 1) / (2 * num_actions))
        # order goes from gamma_k to gamma_1
        self.bases = np.eye(num_actions).tolist()
        self.gammas = np.array(env_params["gammas"])
        gamma_0 = env_params["gamma_0"]
        self.beta_0 = env_params["beta_0"]
        self.beta_1 = env_params["beta_1"]
        self.beta_tilde = lambda a: self.beta_0[a] + self.beta_1[a] * gamma_0
        self.reward_and_kappa = lambda j: self.gammas[j] * np.concatenate((1 / self.beta_1[0], 1 / self.beta_1[1], self.beta_0[0] / self.beta_1[0], self.beta_0[1] / self.beta_1[1]), axis=None)
        self.theta_hat = np.hstack([np.concatenate((self.beta_tilde(a), self.beta_1[int(a)] * np.array([self.reward_and_kappa(j) for j in range(self.k)])), axis=None) for a in range(num_actions)])
        assert self.theta_hat.shape == (num_actions * (2 * self.k * num_actions + 1), )
        self.reward_means = [0] * self.k

    def process_states(self, env, actions, rewards):
        K = env.K
        t = env.get_t()
        recent_actions = actions[t - K:t]
        recent_rewards = rewards[t - K:t]
        predicted_mean_rewards = np.array(self.reward_means[-K:])
        predicted_noises = recent_rewards - predicted_mean_rewards
        predicted_noises[abs(predicted_noises) > 1] = 1 * np.sign(predicted_noises[abs(predicted_noises) > 1])
        reward_diffs = recent_rewards - predicted_noises
        action_states = np.empty((self.num_actions, 2 * self.num_actions * K + 1))
        for a in range(self.num_actions):
            context = np.concatenate([np.concatenate((reward_diff * np.array(self.bases[int(recent_a)]), -1.0 * np.array(self.bases[int(recent_a)])), axis=None) for recent_a, reward_diff in zip(recent_actions, reward_diffs)], axis=None)
            action_states[a] = np.insert(context, 0, 1)
        assert np.array_equal(action_states[0], action_states[1])
        action_set = context_to_action_states(action_states, self.num_actions)
        return action_set

    def select_action(self, states):
        mean_rewards = [state.T @ self.theta_hat for state in states]
        chosen_arm = np.argmax(mean_rewards)
        self.reward_means.append(mean_rewards[chosen_arm])

        return chosen_arm
    
    def update(self, actions, rewards, action_sets, t):
        return None

### Oracle that knows k, the ground-truth parameters, and the mean of the latent process
### but does not get observations of the latent process 
class NonStatOracleAlt(UCBAgent):
    def __init__(self, env_params, state_dim, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__("non_stat_oracle", state_dim, alpha, lambda_reg, num_actions)
        self.k = int((state_dim - 1) / (2 * num_actions))
        # order goes from gamma_k to gamma_1
        self.bases = np.eye(num_actions).tolist()
        self.gammas = np.array(env_params["gammas"])
        gamma_0 = env_params["gamma_0"]
        self.beta_0 = env_params["beta_0"]
        self.beta_1 = env_params["beta_1"]
        self.beta_tilde = lambda a: self.beta_0[a] + self.beta_1[a] * gamma_0
        self.reward_and_kappa = lambda j: self.gammas[j] * np.concatenate((1 / self.beta_1[0], 1 / self.beta_1[1], self.beta_0[0] / self.beta_1[0], self.beta_0[1] / self.beta_1[1]), axis=None)
        self.theta_hat = np.hstack([np.concatenate((self.beta_tilde(a), self.beta_1[int(a)] * np.array([self.reward_and_kappa(j) for j in range(self.k)])), axis=None) for a in range(num_actions)])
        assert self.theta_hat.shape == (num_actions * (2 * self.k * num_actions + 1), )

    def process_states(self, env, actions, rewards):
        K = env.K
        t = env.get_t()
        recent_actions = actions[t - K:t]
        recent_rewards = rewards[t - K:t]
        recent_reward_diffs = recent_rewards - np.array([env.get_reward_noise(t - (K - j)) for j in range(K)])
        should_be_prev_z = [(recent_reward_diffs[i] - self.beta_0[int(action)]) / self.beta_1[int(action)] for action, i in zip(recent_actions, range(len(recent_actions)))]
        assert np.allclose(should_be_prev_z, env.get_all_zs()[t - K:t])
        action_states = np.empty((self.num_actions, 2 * self.num_actions * K + 1))
        for a in range(self.num_actions):
            context = np.concatenate([np.concatenate((reward_diff * np.array(self.bases[int(recent_a)]), -1.0 * np.array(self.bases[int(recent_a)])), axis=None) for recent_a, reward_diff in zip(recent_actions, recent_reward_diffs)], axis=None)
            action_states[a] = np.insert(context, 0, 1)
        assert np.array_equal(action_states[0], action_states[1])
        action_set = context_to_action_states(action_states, self.num_actions)
        # print("TRUE MEAN REWARD!", (env.get_noiseless_reward(0, t), env.get_noiseless_reward(1, t)))
        return action_set
    
    def select_action(self, states):
        mean_rewards = [state.T @ self.theta_hat for state in states]
        chosen_arm = np.argmax(mean_rewards)
        return chosen_arm
    
    # oracle is given the ground-truth parameters and do not need to update
    def update(self, actions, rewards, action_sets, t):
        return None
    
    def get_theta_hat(self):
        return self.theta_hat

