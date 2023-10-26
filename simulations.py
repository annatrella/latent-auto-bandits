from environment import *
from agents import *
import numpy as np
import json
import os
from scipy.stats import bernoulli

GAMMA_SEED = 2023
EXP_SEED = 123
### TRUE ENV. PARAMS ###
K = 5
np.random.seed(GAMMA_SEED)
# need gammas to be between -1 and 1 or else things will blow up
gammas = np.clip(np.random.randn(K), -1, 1)
print("GAMMAS", gammas)  
env_params = {
    "K": K,
    "noise_var": 1e-3,
    "gamma_0": 0.2,
    "gammas": gammas,
    "beta_0": [1.0, 2.0],
    "beta_1": [1.0, 2.0],
    "init_zs": np.zeros(K)
}

### AGENT PARAMS ###
c = 10 # weakly informative
stat_prior_mean = 0
stat_prior_sd = c
non_stat_prior_mean = np.zeros(K + 1)
non_stat_prior_var = c * np.identity(K + 1)
oracle_prior_mean = np.zeros(2)
oracle_prior_var = c * np.identity(2)

STAT_AGENT = StationaryAgent(stat_prior_mean, stat_prior_sd)
NAIVE_NON_STAT_AGENT = NaiveNlaTS(non_stat_prior_mean, non_stat_prior_var)
ORACLE = Oracle(oracle_prior_mean, oracle_prior_var)

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
        noise_var = env.get_noise_var()
        # we do not update with first K time-steps
        agent.update(actions[K:t], rewards[K:t], states[K:t], noise_var)
        ### increment t ###  
        env.increment_t()
    
    return actions, rewards

def calculate_ground_truth(env, seed):
    np.random.seed(seed)
    T = env.get_T()
    ground_truth = {
        "reward 1": np.empty(T),
        "reward 0": np.empty(T),
        "optimal action": np.empty(T)
    }
    while env.get_t() < T:
            ### environment ###
            t = env.get_t()
            print(f"Time Step: {t}")
            env.state_evolution()
            ### get ground truth rewards reward ###
            ground_truth["reward 1"][t] = env.get_noiseless_reward(1)
            ground_truth["reward 0"][t] = env.get_noiseless_reward(0)
            # which action is best for this time-step t
            ground_truth["optimal action"][t] = int(ground_truth["reward 1"][t] > ground_truth["reward 0"][t])
            ### increment t ###  
            env.increment_t()

    return ground_truth

NUM_TIME_STEPS = 100
ground_truth = calculate_ground_truth(Environment(env_params, T=NUM_TIME_STEPS), EXP_SEED)
stat_actions, stat_rewards = run_simulation(Environment(env_params, T=NUM_TIME_STEPS), STAT_AGENT, EXP_SEED)
non_stat_actions, non_stat_rewards = run_simulation(Environment(env_params, T=NUM_TIME_STEPS), NAIVE_NON_STAT_AGENT, EXP_SEED)
oracle_actions, oracle_rewards = run_simulation(Environment(env_params, T=NUM_TIME_STEPS), ORACLE, EXP_SEED)

RESULTS = {
    "stationary_agent": {
        "actions": stat_actions,
        "rewards": stat_rewards
    },
    "naive_non_stat": {
        "actions": non_stat_actions,
        "rewards": non_stat_rewards
    },
    "oracle": {
        "actions": oracle_actions,
        "rewards": oracle_rewards
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

# file_path = os.path.join(directory, f"results_{EXP_SEED}.json")
# with open(file_path , 'w') as json_file:
#     json.dump(RESULTS, json_file, default=convert_to_json_serializable)

