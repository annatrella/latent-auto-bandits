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
EXP_ROOT_FOLDER = "../experiment_results/known_k/"
OUTPUT_FOLDER = "known_k/"
RESULTS_FILE = "/results_{}.json"
GROUND_TRUTH_FILE = "/ground_truth_{}.json"
EXP_PATTERN = re.compile(r'K:(\d+)_noise_std:(\d+(\.\d+)?)')

def list_subfolders(folder_path):
    items = os.listdir(folder_path)
    subfolders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

    return subfolders

def open_json_file(file_name):
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    return data

def get_reward_metrics_for_single_trial(rewards):
    cummulative_rewards = [np.sum(rewards[:t]) for t in range(T)]
    # average_rewards = [np.mean(rewards[:t+1]) for t in range(T)]

    return cummulative_rewards

def get_exp_results_for_single_trial(exp_data, ground_truth):
    all_rewards_metrics = []
    for agent_name, _ in exp_data.items():
        rewards = exp_data[agent_name]["rewards"]
        all_rewards_metrics.append(get_reward_metrics_for_single_trial(rewards))

    optimal_rewards = ground_truth["optimal reward"]    
    all_rewards_metrics.append(get_reward_metrics_for_single_trial(optimal_rewards))

    assert np.array(all_rewards_metrics).shape == (len(exp_data.items()) + 1, T)

    return np.array(all_rewards_metrics)


plt.rcParams.update({
    "text.usetex": True
})
# avg_rewards should be of shape ANNA TODO
def create_and_save_reward_fig(agent_names, avg_rewards, k, sigma_std, exp_name):
    plt.figure(figsize=(8, 6))
    k = int(k)
    for i, agent in enumerate(agent_names):
        mean_value = np.mean(avg_rewards[:,i], axis=0)[k:]
        plt.plot(range(k, T), mean_value, label=agent)
        lower_ci = mean_value + 1.0 * np.std(avg_rewards[:,i], axis=0)[k:]
        upper_ci = mean_value - 1.0 * np.std(avg_rewards[:,i], axis=0)[k:]
        plt.fill_between(range(k, T), lower_ci, upper_ci, alpha=0.2)
    mean_value =  np.mean(avg_rewards[:,-1], axis=0)[k:]
    lower_ci = mean_value + 1.0 * np.std(avg_rewards[:, -1], axis=0)[k:]
    upper_ci = mean_value - 1.0 * np.std(avg_rewards[:, -1], axis=0)[k:]
    plt.plot(range(k, T), mean_value, label='optimal')
    plt.fill_between(range(k, T), lower_ci, upper_ci, alpha=0.2)
    plt.title(f"Cumulative Reward Over Time for \n $k = {k}$, $\sigma_z = {sigma_std}$", fontsize=25)
    plt.xlabel("Time-Step", fontsize=20)
    plt.ylabel("Rewards", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16, loc='best')
    plt.savefig(OUTPUT_FOLDER + f"{exp_name}_cumm_reward_plot.pdf", format='pdf', bbox_inches='tight')
    plt.close()

### Note: assumes that the "oracle" is the last idx in avg_rewards ###
def create_and_save_regret_fig(agent_names, avg_rewards, k, sigma_std, exp_name):
    plt.figure(figsize=(8, 6))
    k = int(k)
    for i, agent in enumerate(agent_names):
        regret_value = np.mean(avg_rewards[:,-1] - avg_rewards[:,i], axis=0)[k:]
        plt.plot(range(k, T), regret_value, label=agent)
        lower_ci = regret_value + 1.0 * np.std(avg_rewards[:,i], axis=0)[k:]
        upper_ci = regret_value - 1.0 * np.std(avg_rewards[:,i], axis=0)[k:]
        plt.fill_between(range(k, T), lower_ci, upper_ci, alpha=0.2)
    plt.plot(range(k, T), np.mean(avg_rewards[:,-1] - avg_rewards[:,-1], axis=0)[k:], label='oracle')
    plt.title(f"Cumulative Regret Over Time for \n $k = {k}$, $\sigma_z = {sigma_std}$", fontsize=25)
    plt.xlabel("Time-Step", fontsize=20)
    plt.ylabel("Regret", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16, loc='best')
    plt.savefig(OUTPUT_FOLDER + f"{exp_name}_cumm_regret_plot.pdf", format='pdf', bbox_inches='tight')
    plt.close()

def get_all_exp_results(parent_folder):
    subfolder = list_subfolders(parent_folder)
    for exp in subfolder:
        subfolder_path = os.path.join(parent_folder, exp)
        exp_data = open_json_file(subfolder_path + RESULTS_FILE.format(0))
        agents = [key for key, _ in exp_data.items()]
        # ground-truth results appended at the end
        exp_reward_metrics = np.zeros((MAX_SEED, len(agents) + 1, T))
        for exp_seed in range(MAX_SEED):
            exp_data = open_json_file(subfolder_path + RESULTS_FILE.format(exp_seed))
            ground_truth = open_json_file(subfolder_path + GROUND_TRUTH_FILE.format(exp_seed))
            exp_reward_metrics[exp_seed] = get_exp_results_for_single_trial(exp_data, ground_truth)

        print(f"Saving Results for Exp: {exp}")
        matches = EXP_PATTERN.match(exp)
        k, sigma_std = matches.group(1), matches.group(2)
        print(f"Making reward graph")
        create_and_save_reward_fig(agents, exp_reward_metrics, k, sigma_std, exp)
        print(f"Making regret graph")
        create_and_save_regret_fig(agents[:-1], exp_reward_metrics[:, :-1, :], k, sigma_std, exp)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
get_all_exp_results(EXP_ROOT_FOLDER)

# need to change this for the true K
# K = 2
# stat_cumm_rewards = np.zeros((MAX_SEED, T))
# non_stat_cumm_rewards = np.zeros((MAX_SEED, T))
# weighted_non_stat_cumm_rewards = np.zeros((MAX_SEED, T))
# oracle_cumm_rewards = np.zeros((MAX_SEED, T))
# optimal_cumm_rewards = np.zeros((MAX_SEED, T))

# stat_avg_rewards = np.zeros((MAX_SEED, T))
# non_stat_avg_rewards = np.zeros((MAX_SEED, T))
# weighted_non_stat_avg_rewards = np.zeros((MAX_SEED, T))
# oracle_avg_rewards = np.zeros((MAX_SEED, T))
# optimal_avg_rewards = np.zeros((MAX_SEED, T))

# for exp_seed in range(MAX_SEED):
#     data = open_json_file(results_file.format(exp_seed))
#     ground_truth = open_json_file(ground_truth_file.format(exp_seed))
#     stat_cumm_rewards[exp_seed], stat_avg_rewards[exp_seed] = get_reward_metrics_for_single_trial(data["stationary_agent"]["rewards"])
#     non_stat_cumm_rewards[exp_seed], non_stat_avg_rewards[exp_seed] = get_reward_metrics_for_single_trial(data["naive_non_stat"]["rewards"])
#     weighted_non_stat_cumm_rewards[exp_seed], weighted_non_stat_avg_rewards[exp_seed] = get_reward_metrics_for_single_trial(data["weighted_non_stat"]["rewards"])
#     oracle_cumm_rewards[exp_seed], oracle_avg_rewards[exp_seed] = get_reward_metrics_for_single_trial(data["non_stat_oracle"]["rewards"])
#     optimal_cumm_rewards[exp_seed], optimal_avg_rewards[exp_seed] = get_reward_metrics_for_single_trial(ground_truth["optimal reward"])

# # AVERAGE
# plt.figure(figsize=(8, 6))
# plt.plot(range(T), np.mean(stat_avg_rewards, axis=0), label='stationary')
# plt.plot(range(T), np.mean(non_stat_avg_rewards, axis=0), label='naive non-stat')
# plt.plot(range(T), np.mean(weighted_non_stat_avg_rewards, axis=0), label='our algorithm')
# plt.plot(range(T), np.mean(oracle_avg_rewards, axis=0), label='oracle')
# # plt.plot(range(T), np.mean(optimal_avg_rewards, axis=0), label='optimal')
# # plt.title("Avg. Cumulative Regret Across Time", fontsize=18)
# plt.xlabel("Time-Step", fontsize=16)
# plt.ylabel("Cumulative Rewards", fontsize=16)
# plt.grid(True)
# plt.legend(fontsize=16)
# plt.savefig("cumm_reward_plot.pdf", format='pdf', bbox_inches='tight')
# plt.close()

# # CUMMULATIVE
# plt.figure(figsize=(8, 6))
# plt.plot(range(T), np.mean(stat_cumm_rewards, axis=0), label='stationary')
# plt.plot(range(T), np.mean(non_stat_cumm_rewards, axis=0), label='naive non-stat')
# plt.plot(range(T), np.mean(weighted_non_stat_cumm_rewards, axis=0), label='our algorithm')
# # plt.plot(range(T), np.mean(oracle_cumm_rewards, axis=0), label='oracle')
# plt.plot(range(T), np.mean(optimal_cumm_rewards, axis=0), label='optimal')
# # plt.title("Avg. Cumulative Regret Across Time", fontsize=18)
# plt.xlabel("Time-Step", fontsize=16)
# plt.ylabel("Average Rewards", fontsize=16)
# plt.grid(True)
# plt.legend(fontsize=16)
# plt.savefig("avg_reward_plot.pdf", format='pdf', bbox_inches='tight')
# plt.close()