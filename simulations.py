from environment import *
from ucb_agents import StationaryAgent, NaiveNSLAR, NonStatOracle
import numpy as np
import json
import os
from scipy.stats import bernoulli

GAMMA_SEED = 2023
EXP_SEED = 234
### TRUE ENV. PARAMS ###
K = 5
np.random.seed(GAMMA_SEED)
# need gammas to be between -1 and 1 or else things will blow up
# gammas = np.clip(np.random.randn(K), -1, 0.999)
gammas = np.array([0.3, 0.1, -0.4, -0.2, 0.5])
print("GAMMAS", gammas)  
env_params = {
    "K": K,
    "noise_var": 1e-3,
    # "noise_var": 1,
    "gamma_0": 0.2,
    "gammas": gammas,
    "beta_0": [0.25, -0.25],
    "beta_1": [1.0, 1.5],
    "init_zs": np.zeros(K)
}

### AGENT PARAMS ###

STAT_AGENT = StationaryAgent()
NAIVE_NON_STAT_AGENT = NaiveNSLAR(K + 1)
NON_STAT_ORACLE = NonStatOracle(env_params, K + 1)

def run_simulation(env, agent, seed):
    np.random.seed(seed)
    T = env.get_T()
    K = env.K
    actions = np.empty(T)
    rewards = np.empty(T)
    states = np.empty((T, agent.get_state_dim()))
    # initializing the first K values
    for t in range(K):
        actions[t] = bernoulli.rvs(0.5)
        rewards[t] = env.get_reward(int(actions[t]))

    while env.get_t() < T:
        ### environment ###
        t = env.get_t()
        print(f"Time Step: {t}")
        env.state_evolution()
        ### action selection ###
        states[t] = agent.process_state(env, rewards)
        actions[t] = agent.select_action(states[t])
        ### produce reward ###
        rewards[t] = env.get_reward(int(actions[t]))
        ### update ###
        # we do not update with first K time-steps
        agent.update(actions[t], rewards[t], states[t])
        ### increment t ###  
        env.increment_t()
    
    return actions, rewards

def calculate_ground_truth(env, seed):
    np.random.seed(seed)
    T = env.get_T()
    ground_truth = {
        "mean reward 1": np.empty(T),
        "mean reward 0": np.empty(T),
        "optimal action": np.empty(T),
        "optimal reward": np.empty(T)
    }
    for t in range(K):
        ground_truth["mean reward 1"][t] = env.get_noiseless_reward(1)
        ground_truth["mean reward 0"][t] = env.get_noiseless_reward(0)
        assert ground_truth["mean reward 1"][t] - ground_truth["mean reward 0"][t] <= 1
        assert  ground_truth["mean reward 0"][t] - ground_truth["mean reward 1"][t] <= 1
        optimal_action = int(ground_truth["mean reward 1"][t] > ground_truth["mean reward 0"][t])
        ground_truth["optimal action"][t] = optimal_action
        reward = env.get_reward(optimal_action)
        ground_truth["optimal reward"][t] = reward        
    while env.get_t() < T:
            ### environment ###
            t = env.get_t()
            print(f"Time Step: {t}")
            env.state_evolution()
            ### get ground truth rewards reward ###
            ground_truth["mean reward 1"][t] = env.get_noiseless_reward(1)
            ground_truth["mean reward 0"][t] = env.get_noiseless_reward(0)
            assert ground_truth["mean reward 1"][t] - ground_truth["mean reward 0"][t] <= 1
            assert  ground_truth["mean reward 0"][t] - ground_truth["mean reward 1"][t] <= 1
            # which action is best for this time-step t
            optimal_action = int(ground_truth["mean reward 1"][t] > ground_truth["mean reward 0"][t])
            ground_truth["optimal action"][t] = optimal_action
            # generate reward
            reward = env.get_reward(optimal_action)
            ground_truth["optimal reward"][t] = reward
            ### increment t ###  
            env.increment_t()

    return ground_truth

NUM_TIME_STEPS = 100
ground_truth = calculate_ground_truth(Environment(env_params, T=NUM_TIME_STEPS), EXP_SEED)
stat_actions, stat_rewards = run_simulation(Environment(env_params, T=NUM_TIME_STEPS), STAT_AGENT, EXP_SEED)
non_stat_actions, non_stat_rewards = run_simulation(Environment(env_params, T=NUM_TIME_STEPS), NAIVE_NON_STAT_AGENT, EXP_SEED)
non_stat_oracle_actions, non_stat_oracle_rewards = run_simulation(Environment(env_params, T=NUM_TIME_STEPS), NON_STAT_ORACLE, EXP_SEED)

RESULTS = {
    "stationary_agent": {
        "actions": stat_actions,
        "rewards": stat_rewards
    },
    "naive_non_stat": {
        "actions": non_stat_actions,
        "rewards": non_stat_rewards
    },
    "non_stat_oracle": {
        "actions": non_stat_oracle_actions,
        "rewards": non_stat_oracle_rewards
    }
}

### SAVING RESULTS ###
# Convert NumPy arrays to lists and save the dictionary as a JSON file
def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"{type(obj)} is not JSON serializable")
    
def save_to_json(file_path, json_result):
    with open(file_path , 'w') as json_file:
        json.dump(json_result, json_file, default=convert_to_json_serializable)
    print(f"Results have been saved as {file_path}")
    
directory = "experiment_results"
if not os.path.exists(directory):
    os.makedirs(directory)

save_to_json(os.path.join(directory, f"results_{EXP_SEED}.json"), RESULTS)
save_to_json(os.path.join(directory,f"ground_truth_{EXP_SEED}.json"), ground_truth)

