from simulations import *
from generate_valid_env import generate_centered_stable_weights
from ucb_agents import StationaryAgent, NonStatRLS, NonStatOracleAlt
from global_params import MAX_SEED, NUM_TIME_STEPS

import itertools

### TRUE ENV. PARAMS ###
TRUE_K = 5
TRUE_GAMMAS = generate_centered_stable_weights(TRUE_K, 2, 123)
K_0s = [1, 3, 6, 10]

PARAMS = dict(gamma_0=[0],
        beta_0=[[0,0]],
        beta_1=[[-1.0, 1.0]],
        noise_std=[1],
        K_0=K_0s
        )
DICT_KEYS = ['gamma_0', 'beta_0', 'beta_1', 'noise_std', 'K_0', 'K', 'gammas', 'init_zs']
OUTPUT_PATH_NAMES = ['K_0']
EXPERIMENTS = {}

for vals in itertools.product(*list(PARAMS.values())):
    exp_values = list(vals[:-1])
    # K_0
    exp_values.append(vals[-1])
    exp_values.append(TRUE_K)
    exp_values.append(TRUE_GAMMAS)
    exp_kwargs = dict(zip(DICT_KEYS, exp_values))
    exp_name = "_".join([f"{key}:{exp_kwargs[key]}" for key in OUTPUT_PATH_NAMES])
    EXPERIMENTS[exp_name] = exp_kwargs

LAMBDA_REG = 0.1
STAT_AGENT = lambda k, env_params: StationaryAgent(TRUE_K)
OUR_ALGORITHM = lambda k, env_params: NonStatRLS(2 * 2 * TRUE_K + 1)
WRONG_K_ALGORITHM = lambda k, env_params: NonStatRLS(2 * 2 * k + 1)
NON_STAT_ORACLE = lambda k, env_params: NonStatOracleAlt(env_params, 2 * 2 * TRUE_K + 1)

AGENTS = [STAT_AGENT, OUR_ALGORITHM, WRONG_K_ALGORITHM, NON_STAT_ORACLE]
AGENT_NAMES = ['stationary', 'ours (true k)', 'ours (wrong k)', 'oracle']

def run_experiment(exp_name, env_params, agents):
    print("ENV PARAMS!", env_params)
    for exp_seed in range(0, MAX_SEED):
        RESULTS = {}
        # we draw new init z's every seed
        env_params['init_zs'] = np.random.randn(env_params['K'])
        ground_truth = calculate_ground_truth(Environment(env_params, T=NUM_TIME_STEPS), exp_seed)
        for agent, agent_name in zip(agents, AGENT_NAMES):
            agent = agent(env_params['K_0'], env_params)
            actions, rewards = run_simulation(Environment(env_params, T=NUM_TIME_STEPS), agent, exp_seed)
            RESULTS[agent_name] = {
                "actions": actions,
                "rewards": rewards
            }
        directory = f"experiment_results/unknown_k/{exp_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        save_to_json(os.path.join(directory, f"results_{exp_seed}.json"), RESULTS)
        save_to_json(os.path.join(directory,f"ground_truth_{exp_seed}.json"), ground_truth)

for name, env_params in EXPERIMENTS.items():
    print(f"Starting experiment: {name}")
    run_experiment(name, env_params, AGENTS)