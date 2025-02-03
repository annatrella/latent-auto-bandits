import numpy as np
import json
from scipy.stats import bernoulli

### AGENT PARAMS ###

def run_simulation(env, agent, seed):
    np.random.seed(seed)
    T = env.get_T()
    K = env.K
    actions = np.zeros(T)
    rewards = np.zeros(T)
    states = np.zeros((T, agent.get_state_dim()))
    # initializing the first K values
    for _ in range(K):
        t = env.get_t()
        states[t] = agent.process_state(env, actions, rewards)
        actions[t] = agent.select_action(env, states[t])
        rewards[t] = env.get_reward(int(actions[t]))
        env.increment_t()

    while env.get_t() < T:
        ### environment ###
        t = env.get_t()
        # print(f"Time Step: {t}")
        env.state_evolution()
        ### action selection ###
        states[t] = agent.process_state(env, actions, rewards)
        actions[t] = agent.select_action(env, states[t])
        ### produce reward ###
        rewards[t] = env.get_reward(int(actions[t]))
        ### update ###
        # we do not update with first K time-steps
        agent.update(actions, rewards, states, t)
        ### increment t ###  
        env.increment_t()
    
    return actions, rewards, states

def calculate_ground_truth(env, seed):
    np.random.seed(seed)
    T = env.get_T()
    K = env.K
    ground_truth = {
        "k": env.K,
        "sigma_z": env.sigma_z, 
        "mean reward 1": np.zeros(T),
        "mean reward 0": np.zeros(T),
        "optimal action": np.zeros(T),
        "optimal reward": np.zeros(T),
        "zs": np.zeros(T)
    }
    for _ in range(K):
        t = env.get_t()
        ground_truth["mean reward 1"][t] = env.get_noiseless_reward(1, t)
        ground_truth["mean reward 0"][t] = env.get_noiseless_reward(0, t)
        optimal_action = int(ground_truth["mean reward 1"][t] > ground_truth["mean reward 0"][t])
        ground_truth["optimal action"][t] = optimal_action
        ground_truth["zs"][t] = env.get_all_zs()[t]
        reward = env.get_reward(optimal_action)
        ground_truth["optimal reward"][t] = reward
        env.increment_t()    
    while env.get_t() < T:
            ### environment ###
            t = env.get_t()
            # print(f"Time Step: {t}")
            env.state_evolution()
            ### get ground truth rewards reward ###
            ground_truth["mean reward 1"][t] = env.get_noiseless_reward(1, t)
            ground_truth["mean reward 0"][t] = env.get_noiseless_reward(0, t)
            # which action is best for this time-step t
            optimal_action = int(ground_truth["mean reward 1"][t] > ground_truth["mean reward 0"][t])
            ground_truth["optimal action"][t] = optimal_action
            # save latent state z
            ground_truth["zs"][t] = env.get_all_zs()[t]
            # generate reward
            reward = env.get_reward(optimal_action)
            ground_truth["optimal reward"][t] = reward
            ### increment t ###  
            env.increment_t()

    return ground_truth

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


