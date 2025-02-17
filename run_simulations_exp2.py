import os
import numpy as np
from environment import Environment
from simulations import run_simulation, calculate_ground_truth, save_to_json
from generate_valid_env import generate_centered_stable_weights
from ucb_agents import StationaryAgent, LatentARLinUCB, ARUCB, SWUCB
from baseline_non_stat_agents import Rexp3
from global_params import MAX_SEED, NUM_TIME_STEPS

import itertools

### TRUE ENV. PARAMS ###
Ks = [1, 5, 10]
GAMMAS = []
for k in Ks:
    if k == 1:
        gammas = [0.9]
    else:
        # need gammas to be between -1 and 1 or else things will blow up
        gammas = generate_centered_stable_weights(k, 2, 123)
    print("GAMMAS", gammas)  
    GAMMAS.append(gammas)

PARAMS = dict(gamma_0=[0],
        mu_a=[[0,0]],
        beta_a=[[-1.0, 1.0]],
        sigma_z=[1],
        sigma_r=[1],
        k_index=[i for i in range(len(Ks))]
        )
DICT_KEYS = ['gamma_0', 'mu_a', 'beta_a', 'sigma_z', 'sigma_r', 'K', 'gammas', 'init_zs']
OUTPUT_PATH_NAMES = ['K', 'sigma_z', 'sigma_r']
EXPERIMENTS = {}

for vals in itertools.product(*list(PARAMS.values())):
    exp_values = list(vals[:-1])
    k_index = vals[-1]
    exp_values.append(Ks[k_index])
    exp_values.append(GAMMAS[k_index])
    exp_kwargs = dict(zip(DICT_KEYS, exp_values))
    exp_name = "_".join([f"{key}:{exp_kwargs[key]}" for key in OUTPUT_PATH_NAMES])
    EXPERIMENTS[exp_name] = exp_kwargs

def run_experiment(exp_name, env_name, env_params, agents):
    for exp_seed in range(0, MAX_SEED):
        RESULTS = {}
        # we draw new init z's every seed
        env_params['init_zs'] = 10 * np.random.randn(env_params['K'])
        # agents += [KALMAN_FILTER_AGENT(0, env_params)]
        ground_truth = calculate_ground_truth(Environment(env_params, T=NUM_TIME_STEPS), exp_seed)
        for agent in agents:
            actions, rewards, _ = run_simulation(Environment(env_params, T=NUM_TIME_STEPS), agent, exp_seed)
            RESULTS[agent.name] = {
                "actions": actions,
                "rewards": rewards
            }
        directory = f"experiment_results/{exp_name}/{env_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        save_to_json(os.path.join(directory, f"results_{exp_seed}.json"), RESULTS)
        save_to_json(os.path.join(directory,f"ground_truth_{exp_seed}.json"), ground_truth)

for env_name, env_params in EXPERIMENTS.items():
    # READ ME: to run a different experiment, please first chage exp_name and then 
    # modify the agents, environment, etc.
    exp_name = "against_baselines"
    print(f"Starting experiment: {exp_name} {env_name}")

    AGENTS = [StationaryAgent(), LatentARLinUCB(env_params['K']), ARUCB(env_params['K']), SWUCB(env_params['K']), Rexp3(NUM_TIME_STEPS, NUM_TIME_STEPS)]
    
    run_experiment(exp_name, env_name, env_params, AGENTS)