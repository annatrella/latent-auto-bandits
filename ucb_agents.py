import numpy as np
from rls import fit_rls

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

    def process_state(self, env, actions, rewards):
        return None

    # upper confidence provided by thie article: https://www.linkedin.com/pulse/contextual-bandits-linear-upper-confidence-bound-disjoint-kenneth-foo/
    def select_action(self, env, state):
        # print(f"State: {state}")
        # print(f"VS!: {self.Vs}")
        # Compute the optimistic estimates for each arm
        optimism = [self.alpha * np.sqrt(np.dot(state.T, np.dot(np.linalg.inv(V), state)))
                     for V in self.Vs]
        optimistic_estimates = [np.dot(theta_hat.T, state) + optimism_value
                                for theta_hat, optimism_value in zip(self.theta_hats, optimism)]

        # Choose the arm with the maximum optimistic estimate
        chosen_action = np.argmax(optimistic_estimates)
        return chosen_action
    
    def update(self, actions, rewards, states, t):
        # print(f"actions: {actions}")
        action = int(actions[t])
        reward = rewards[t]
        state = states[t]
        # Update V and b for the chosen arm using Bayesian linear regression
        self.Vs[action], self.bs[action], self.theta_hats[action] = fit_rls(self.Vs[action], self.bs[action], state, reward)
    
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
    def __init__(self, num_actions=2):
        super().__init__("Stationary", 1, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=num_actions)

    def process_state(self, env, actions, rewards):
        return np.ones(1)
    
class LatentARLinUCB(UCBAgent):
    def __init__(self, s, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__("Latent AR LinUCB", 2 * s * num_actions + 1, alpha, lambda_reg, num_actions)
        self.s = s
        # order goes from 
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

### Intermediate agent that knows the ground-truth parameters and runs a standard Kalman filter
### but does not get observations of the latent process 
class IntermediateAgent(UCBAgent):
    def __init__(self, env_params, s, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__("intermediate", 2 * s * num_actions + 1, alpha, lambda_reg, num_actions)
        self.s = s
        # ANNA TODO: need to fix this
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

    def process_state(self, env, actions, rewards):
        t = env.get_t()
        s = self.s
        recent_actions = actions[t - s:t]
        recent_rewards = rewards[t - s:t]
        X_t = X_t = np.array([reward * np.array(self.bases[action]) for reward, action in zip(recent_rewards, recent_actions)]).flatten()
        A_t = np.array([self.bases[action] for action in recent_actions]).flatten()

        return np.concatenate([X_t, A_t, [1]])
    
    def select_action(self, env, state):
        mean_rewards = [state.T @ self.theta_hat for state in states]
        chosen_arm = np.argmax(mean_rewards)
        return chosen_arm
    
    # oracle is given the ground-truth parameters and do not need to update
    def update(self, actions, rewards, action_sets, t):
        return None
    
    def get_theta_hat(self):
        return self.theta_hat

class UniformRandom(LatentARLinUCB):
    def __init__(self, state_dim, alpha=ALPHA, lambda_reg=LAMBDA_REG, num_actions=2):
        super().__init__(state_dim, alpha, lambda_reg, num_actions)

    def select_action(self, env, state):
        return np.random.choice([0, 1], size=1, p=[0.5, 0.5])
    
    def update(self, actions, rewards, action_sets, t):
        return None