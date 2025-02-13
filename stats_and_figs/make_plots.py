import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import re

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from global_params import MAX_SEED, NUM_TIME_STEPS

T = NUM_TIME_STEPS
EXP_ROOT_FOLDER = "../experiment_results/varying_s/"
OUTPUT_FOLDER = "varying_s/"
RESULTS_FILE = "/results_{}.json"
GROUND_TRUTH_FILE = "/ground_truth_{}.json"

def extract_values(input_string):
    k_match = re.search(r'K:\s*(\d+)', input_string)
    sigma_z_match = re.search(r'_sigma_z:\s*([+-]?\d*\.\d+|\d+)', input_string)
    sigma_r_match = re.search(r'_sigma_r:\s*([+-]?\d*\.\d+|\d+)', input_string)
    
    k_value = k_match.group(1) if k_match else None
    sigma_z_value = sigma_z_match.group(1) if sigma_z_match else None
    sigma_r_value = sigma_r_match.group(1) if sigma_r_match else None
    
    return k_value, sigma_z_value, sigma_r_value

def list_subfolders(folder_path):
    items = os.listdir(folder_path)
    subfolders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

    return subfolders

def open_json_file(file_name):
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    return data

def actions_to_reward_means(actions, ground_truth):
    T = len(actions)
    result = np.zeros(T)
    for t, action in enumerate(actions):
        result[t] = ground_truth["mean reward 1"][t] if action == 1 else ground_truth["mean reward 0"][t]
    return result

def get_reward_metrics_for_single_trial(actions, ground_truth, k):
    reward_means = actions_to_reward_means(actions, ground_truth)
    cummulative_rewards = [np.sum(reward_means[k:t]) for t in range(k,T)]

    return cummulative_rewards

def get_exp_results_for_single_trial(exp_data, ground_truth):
    k = ground_truth["k"]
    all_rewards_metrics = []
    for agent_name, _ in exp_data.items():
        actions = exp_data[agent_name]["actions"]
        all_rewards_metrics.append(get_reward_metrics_for_single_trial(actions, ground_truth, k))

    optimal_actions = ground_truth["optimal action"]    
    all_rewards_metrics.append(get_reward_metrics_for_single_trial(optimal_actions, ground_truth, k))

    assert np.array(all_rewards_metrics).shape == (len(exp_data.items()) + 1, T - k)

    return np.array(all_rewards_metrics)


plt.rcParams.update({
    "text.usetex": True
})
# cumm_reward_means should be of shape (NUM_TRIALS, NUM_AGENTS, NUM_TIME_STEPS)
def create_and_save_reward_fig(agent_names, cumm_reward_means, k, sigma_z, sigma_r, exp_name):
    plt.figure(figsize=(8, 6))
    k = int(k)
    for i, agent in enumerate(agent_names):
        mean_value = np.mean(cumm_reward_means[:,i], axis=0)
        plt.plot(range(k, T), mean_value, label=agent)
        lower_ci = mean_value + 1.0 * np.std(cumm_reward_means[:,i], axis=0)
        upper_ci = mean_value - 1.0 * np.std(cumm_reward_means[:,i], axis=0)
        plt.fill_between(range(k, T), lower_ci, upper_ci, alpha=0.2)
    mean_value =  np.mean(cumm_reward_means[:,-1], axis=0)
    lower_ci = mean_value + 1.0 * np.std(cumm_reward_means[:, -1], axis=0)
    upper_ci = mean_value - 1.0 * np.std(cumm_reward_means[:, -1], axis=0)
    plt.plot(range(k, T), mean_value, label='Optimal')
    plt.fill_between(range(k, T), lower_ci, upper_ci, alpha=0.2)
    plt.title(f"Cumulative Reward Means Over Time for \n $k = {k}$, $\sigma_z = {sigma_z}$, $\sigma_r = {sigma_r}$", fontsize=25)
    plt.xlabel("Time-Step", fontsize=20)
    plt.ylabel("Mean Rewards", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16, loc='best')
    plt.savefig(OUTPUT_FOLDER + f"/reward_means/{exp_name}_cumm_reward_means_plot.pdf", format='pdf', bbox_inches='tight')
    plt.close()

### Note: assumes that the "oracle" is the last idx in cumm_reward_means ###
def create_and_save_regret_fig(agent_names, cumm_reward_means, k, sigma_z, sigma_r, exp_name):
    plt.figure(figsize=(8, 6))
    k = int(k)
    for i, agent in enumerate(agent_names):
        regret_value = np.mean(cumm_reward_means[:,-1] - cumm_reward_means[:,i], axis=0)
        plt.plot(range(k, T), regret_value, label=agent)
        lower_ci = regret_value + 1.0 * np.std(cumm_reward_means[:,i], axis=0)
        upper_ci = regret_value - 1.0 * np.std(cumm_reward_means[:,i], axis=0)
        plt.fill_between(range(k, T), lower_ci, upper_ci, alpha=0.2)
    plt.title(f"$k = {k}$, $\sigma_z = {sigma_z}$, $\sigma_r = {sigma_r}$", fontsize=25)
    plt.xlabel("Time-Step", fontsize=20)
    plt.ylabel("Cumulative Regret", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16, loc='best')
    plt.savefig(OUTPUT_FOLDER + f"/regret/{exp_name}_cumm_regret_plot.pdf", format='pdf', bbox_inches='tight')
    plt.close()

def get_all_exp_results(parent_folder):
    subfolder = list_subfolders(parent_folder)
    for exp in subfolder:
        k, sigma_z, sigma_r = extract_values(exp)
        subfolder_path = os.path.join(parent_folder, exp)
        exp_data = open_json_file(subfolder_path + RESULTS_FILE.format(0))
        agents = [key for key, _ in exp_data.items()]
        # ground-truth results appended at the end
        exp_reward_metrics = np.zeros((MAX_SEED, len(agents) + 1, T - int(k)))
        for exp_seed in range(MAX_SEED):
            exp_data = open_json_file(subfolder_path + RESULTS_FILE.format(exp_seed))
            ground_truth = open_json_file(subfolder_path + GROUND_TRUTH_FILE.format(exp_seed))
            exp_reward_metrics[exp_seed] = get_exp_results_for_single_trial(exp_data, ground_truth)

        print(f"Saving Results for Exp: {exp}")
        print(f"Making reward graph")
        create_and_save_reward_fig(agents, exp_reward_metrics, k, sigma_z, sigma_r, exp)
        print(f"Making regret graph")
        create_and_save_regret_fig(agents, exp_reward_metrics, k, sigma_z, sigma_r, exp)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER + "reward_means/")
    os.makedirs(OUTPUT_FOLDER + "regret/")
get_all_exp_results(EXP_ROOT_FOLDER)