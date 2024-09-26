import os
import numpy as np
from environment import Environment
from simulations import run_simulation, calculate_ground_truth, save_to_json
from generate_valid_env import generate_centered_stable_weights
from ucb_agents import StationaryAgent, LatentARLinUCB, IntermediateAgent
from global_params import MAX_SEED, NUM_TIME_STEPS

import itertools

### TRUE ENV. PARAMS ###
Ks = [1, 2, 5, 10]
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
        beta_a=[[-100.0, 100.0]],
        noise_std=[1e-3, 1, 10],
        k_index=[i for i in range(len(Ks))]
        )
DICT_KEYS = ['gamma_0', 'mu_a', 'beta_a', 'noise_std', 'K', 'gammas', 'init_zs']
OUTPUT_PATH_NAMES = ['K', 'noise_std']
EXPERIMENTS = {}

for vals in itertools.product(*list(PARAMS.values())):
    exp_values = list(vals[:-1])
    k_index = vals[-1]
    exp_values.append(Ks[k_index])
    exp_values.append(GAMMAS[k_index])
    exp_kwargs = dict(zip(DICT_KEYS, exp_values))
    exp_name = "_".join([f"{key}:{exp_kwargs[key]}" for key in OUTPUT_PATH_NAMES])
    EXPERIMENTS[exp_name] = exp_kwargs

LAMBDA_REG = 0.1
STAT_AGENT = StationaryAgent()
OUR_ALGORITHM = lambda s, env_params: LatentARLinUCB(s)
INTERMEDIATE_AGENT = lambda s, env_params: IntermediateAgent(env_params, s)

AGENTS = [STAT_AGENT, OUR_ALGORITHM(1, None), OUR_ALGORITHM(2, None), OUR_ALGORITHM(5, None), OUR_ALGORITHM(10, None)]
AGENT_NAMES = ['stationary', 'ours']
# AGENTS = [STAT_AGENT, OUR_ALGORITHM, INTERMEDIATE_AGENT]
# AGENT_NAMES = ['stationary', 'ours', 'intermediate']

def run_experiment(exp_name, env_params, agents):
    for exp_seed in range(0, MAX_SEED):
        RESULTS = {}
        # we draw new init z's every seed
        env_params['init_zs'] = np.random.randn(env_params['K'])
        ground_truth = calculate_ground_truth(Environment(env_params, T=NUM_TIME_STEPS), exp_seed)
        for agent in agents:
            actions, rewards, _ = run_simulation(Environment(env_params, T=NUM_TIME_STEPS), agent, exp_seed)
            key_name = agent.name if agent.name == 'Stationary' else f"{agent.name} s={agent.s}"
            RESULTS[key_name] = {
                "actions": actions,
                "rewards": rewards
            }
        directory = f"experiment_results/fixed_s/{exp_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        save_to_json(os.path.join(directory, f"results_{exp_seed}.json"), RESULTS)
        save_to_json(os.path.join(directory,f"ground_truth_{exp_seed}.json"), ground_truth)

for name, env_params in EXPERIMENTS.items():
    print(f"Starting experiment: {name}")
    run_experiment(name, env_params, AGENTS)