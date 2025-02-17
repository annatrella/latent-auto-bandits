import numpy as np
from rls import fit_rls
from lds import create_lds_from_ar_process, compute_P, compute_K, compute_z_tilde

LAMBDA_REG = 1.0
ALPHA = 1.0

### Parent Agent Class ###
class UCBAgent:
    def __init__(self, name, state_dim, alpha, lambda_reg, num_actions=2):
        self.name = name
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.alpha = alpha # exploration parameter from LinUCB
        self.lambda_reg = lambda_reg
        self.Vs = [self.lambda_reg * np.eye(self.state_dim, dtype=np.float64) for _ in range(self.num_actions)]
        self.bs = [np.zeros(self.state_dim, dtype=np.float64) for _ in range(self.num_actions)]
        self.theta_hats = [np.linalg.inv(self.Vs[i]) @ self.bs[i] for i in range(self.num_actions)]
        self.reward_means = []

    def process_state(self, env, actions, rewards):
        return None

    # upper confidence provided by thie article: https://www.linkedin.com/pulse/contextual-bandits-linear-upper-confidence-bound-disjoint-kenneth-foo/
    def select_action(self, env, state):
        # Compute the optimistic estimates for each arm
        optimism = [self.alpha * np.sqrt(np.dot(state.T, np.dot(np.linalg.inv(V), state)))
                     for V in self.Vs]
        optimistic_estimates = [np.dot(theta_hat.T, state) + optimism_value
                                for theta_hat, optimism_value in zip(self.theta_hats, optimism)]
        self.reward_means.append(optimistic_estimates)
        # Choose the arm with the maximum optimistic estimate
        chosen_action = np.argmax(optimistic_estimates)
        return chosen_action
    
    def update(self, actions, rewards, states, t):
        action = int(actions[t])
        reward = rewards[t]
        state = states[t]
        # Update V and b for the chosen arm using Bayesian linear regression
        self.Vs[action], self.bs[action], self.theta_hats[action] = fit_rls(self.Vs[action], self.bs[action], state, reward)
    
    def get_state_dim(self):
        return self.state_dim
    
    def get_theta_hats(self):
        return self.theta_hats
    
    def get_Vs(self):
        return self.Vs
    
    def set_theta_hats(self, theta_hats):
        self.theta_hats = theta_hats
    
### Helpers ###
    
### Standard (Stationary) MAB ###
class StationaryAgent(UCBAgent):
    def __init__(self, num_actions=2):
        super().__init__("Stationary", 1, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=num_actions)

    def process_state(self, env, actions, rewards):
        return np.ones(1)

    def select_action(self, env, state):
        t = env.get_t()
        if t < env.get_k():
            return np.random.choice(range(self.num_actions))
        else:
            return super().select_action(env, state)

### Stationary AR Bandit (Bacchiocchi et. al, 2022) ###        
class ARUCB(UCBAgent):
    def __init__(self, s, num_actions=2):
        super().__init__("AR UCB", s, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=num_actions)
        self.s = s

    def process_state(self, env, actions, rewards):
        t = env.get_t()
        if t < self.s:
            return None
        return np.array(rewards[t - self.s:t])
        
    def select_action(self, env, state):
        t = env.get_t()
        if t < env.get_k():
            return np.random.choice(range(self.num_actions))
        else:
            return super().select_action(env, state)

    
class LatentARLinUCB(UCBAgent):
    def __init__(self, s, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__("LARL", 2 * s * num_actions + 1, alpha, lambda_reg, num_actions)
        self.s = s 
        self.bases = np.eye(num_actions).tolist()

    def process_state(self, env, actions, rewards):
        t = env.get_t()
        s = self.s
        if t < s:
            return None
        else:
            recent_actions = actions[t - s:t]
            recent_rewards = rewards[t - s:t]
            X_t = np.array([reward * np.array(self.bases[int(action)]) for reward, action in zip(recent_rewards, recent_actions)]).flatten()
            A_t = np.array([self.bases[int(action)] for action in recent_actions]).flatten()
            assert len(np.concatenate([X_t, A_t, [1]])) == 2 * s * self.num_actions + 1
            return np.concatenate([X_t, A_t, [1]])

    def select_action(self, env, state):
        t = env.get_t()
        if t < self.s:
            return np.random.choice(range(self.num_actions))
        else:
            return super().select_action(env, state)
        
    def update(self, actions, rewards, states, t):
        if t >= self.s:
            super().update(actions, rewards, states, t)

### KalmanFilter agent that knows the ground-truth parameters and runs a standard Kalman filter
### but does not get observations of the latent process 
class KalmanFilterAgent(UCBAgent):
    def __init__(self, env_params, s, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__(r"Kalman Filter w/ $\theta^*$", 2 * s * num_actions + 1, alpha, lambda_reg, num_actions)
        self.s = s
        self.k = env_params["K"]
        # order goes from gamma_1 to gamma_k
        self.bases = np.eye(num_actions).tolist()
        self.gammas = np.array(env_params["gammas"])
        self.gamma_0 = env_params["gamma_0"]
        self.mu_a = env_params["mu_a"]
        self.beta_a = env_params["beta_a"]
        self.sigma_z = env_params["sigma_z"]
        self.sigma_r = env_params["sigma_r"]
        self.c_a = np.zeros((num_actions , self.k))
        self.c_a[:, 0] = self.beta_a
        self.Gamma, self.C, self.W, V, self.mu_z = create_lds_from_ar_process(self.gammas, self.gamma_0, self.k, self.sigma_z, self.sigma_r, self.beta_a)
        P = compute_P(self.Gamma, self.C, self.W, V)
        self.Kalman_Gain = compute_K(P, self.C, V)
        # z_tilde_t = [\tilde{z}_t \tilde{z}_{t - 1} ... \tilde{z}_{t - k}]
        self.z_tildes = [np.flip(env_params['init_zs'])] * self.k
        self.observations = []
        # for debugging
        self.reward_means = []

    def process_state(self, env, actions, rewards):
        t = env.get_t()
        # for the first time step, there is no previous observation
        if t > 0:
            action = int(actions[t - 1])
            reward = rewards[t - 1]
            prev_y = (reward - self.mu_a[action]) / self.beta_a[action]
            self.observations.append(prev_y)

        return None
    
    def select_action(self, env, state):
        t = env.get_t()
        if t < env.get_k():
            z_t_tilde = self.z_tildes[t]
        else:
            last_z_tilde = self.z_tildes[t - 1]
            last_y = self.observations[t - 1]
            z_t_tilde = compute_z_tilde(self.Gamma, self.Kalman_Gain, self.C, last_z_tilde, self.mu_z, [last_y])
            self.z_tildes.append(z_t_tilde)
        mean_rewards = [c_a.T @ z_t_tilde + mu_a for c_a, mu_a in zip(self.c_a, self.mu_a)]
        self.reward_means.append(mean_rewards)
        chosen_arm = np.argmax(mean_rewards)

        return chosen_arm
    
    # note: intermediate agent only updates observations 
    # and is given the ground-truth parameters and do not need to update parameters
    def update(self, actions, rewards, states, t):

        return None
    
### Sliding Window UCB (Garivier & Moulines, 2008) ###
# uses a sliding window approach of size tau = k
# code adapted from: https://github.com/MaxenceGiraud/ucb-nonstationary/blob/main/nsucb/sliding_ucb.py
class SWUCB(UCBAgent):
    def __init__(self, k, num_actions=2):
        super().__init__("SW UCB", 1, ALPHA, LAMBDA_REG, num_actions)
        self.tau = k
        self.B = 100 # upper bound on rewards
        self.xi = 0.5 # original authors set 0.5 for simulations
        self.optimism_values = np.zeros(self.num_actions)

    def process_states(self, env, actions, rewards):
        return np.ones(1)
    
    def select_action(self, env, state):
        t = env.get_t()
        if t < self.tau:
            return np.random.choice(range(self.num_actions))
        else:
            return np.argmax([optimism_a for optimism_a in self.optimism_values])
    
    def update(self, actions, rewards, states, t):
        for i in range(self.num_actions):
            N = np.sum(actions[-self.tau:])
            X = (1 / N) * np.sum(rewards[-self.tau:])
            c = self.B * np.sqrt((self.xi * np.log(max(t, self.tau))) / N)
            self.optimism_values[i] = X + c