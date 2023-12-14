import json
import matplotlib.pyplot as plt
import numpy as np

results_file = "../experiment_results/results_234.json"
ground_truth_file = "../experiment_results/ground_truth_234.json"

def open_json_file(file_name):
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    return data

data = open_json_file(results_file)
ground_truth = open_json_file(ground_truth_file)

# Extract the episode and reward data
optimal_rewards = ground_truth["optimal reward"]
stat_rewards = data["stationary_agent"]["rewards"]
non_stat_rewards = data["naive_non_stat"]["rewards"]
non_stat_oracle_rewards = data["non_stat_oracle"]["rewards"]
# standard_oracle_rewards = data["standard_oracle"]["rewards"]
T = len(stat_rewards)
# Create a line plot
plt.figure(figsize=(8, 6))
plt.plot(range(T), stat_rewards, marker='o', linestyle='-', label='stationary')
plt.plot(range(T), non_stat_rewards, marker='o', linestyle='-', label='non-stationary')
plt.plot(range(T), non_stat_oracle_rewards, marker='o', linestyle='-', label='oracle')
plt.plot(range(T), optimal_rewards, marker='o', linestyle='-', label='optimal')
plt.title("Reward Over Time", fontsize=18)
plt.xlabel("Time-Step", fontsize=16)
plt.ylabel("Reward", fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.savefig("reward_plot.pdf", format='pdf', bbox_inches='tight')
plt.close()

cummulative_optimal_rewards = [np.sum(optimal_rewards[:t]) for t in range(T)]
cummulative_stat_rewards = [np.sum(stat_rewards[:t]) for t in range(T)]
cummulative_non_stat_rewards = [np.sum(non_stat_rewards[:t]) for t in range(T)]
cummulative_non_stat_oracle_rewards = [np.sum(non_stat_oracle_rewards[:t]) for t in range(T)]
# cumulative reward across time
plt.figure(figsize=(8, 6))
plt.plot(range(T), cummulative_stat_rewards, marker='o', linestyle='-', label='stationary')
plt.plot(range(T), cummulative_non_stat_rewards, marker='o', linestyle='-', label='non-stationary')
plt.plot(range(T), cummulative_non_stat_oracle_rewards, marker='o', linestyle='-', label='oracle')
plt.plot(range(T), cummulative_optimal_rewards, marker='o', linestyle='-', label='optimal')
plt.title("Cumulative Reward Over Time", fontsize=18)
plt.xlabel("Time-Step", fontsize=16)
plt.ylabel("Reward", fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.savefig("cummulative_reward_plot.pdf", format='pdf', bbox_inches='tight')
plt.close()

optimal_actions = ground_truth["optimal action"]
stat_actions = data["stationary_agent"]["actions"]
non_stat_actions = data["naive_non_stat"]["actions"]
non_stat_oracle_actions = data["non_stat_oracle"]["actions"]
cummulative_stat_actions = [np.sum([optimal_actions[i] == stat_actions[i] for i in range(t)]) for t in range(T)]
cummulative_non_stat_actions = [np.sum([optimal_actions[i] == non_stat_actions[i] for i in range(t)]) for t in range(T)]
cummulative_non_stat_oracle_actions = [np.sum([optimal_actions[i] == non_stat_oracle_actions[i] for i in range(t)]) for t in range(T)]
stat_actions_match = [optimal_actions[i] == stat_actions[i] for i in range(T)]
non_stat_actions_match = [optimal_actions[i] == non_stat_actions[i] for i in range(T)]
non_stat_oracle_actions_match = [optimal_actions[i] == non_stat_oracle_actions[i] for i in range(T)]

# comparing to ground truth
plt.figure(figsize=(8, 6))
plt.plot(range(T), cummulative_stat_actions, linestyle='-', label='stationary')
plt.plot(range(T), cummulative_non_stat_actions, linestyle='-', label='non-stationary')
plt.plot(range(T), cummulative_non_stat_oracle_actions, linestyle='-', label='oracle')
plt.plot(range(1, T + 1), range(1, T + 1), color='red', label='optimal')
plt.title("Total Num. Times Chose Optimal Action Up to Time-Step", fontsize=18)
plt.xlabel("Time-Step", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.savefig("cummulative_actions_plot.pdf", format='pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(range(T), stat_actions_match, marker='o', label='stationary')
plt.scatter(range(T), non_stat_actions_match, marker='o', label='non-stationary')
plt.scatter(range(T), non_stat_oracle_actions_match, marker='o', label='oracle')
plt.scatter(range(T), np.ones(T), marker='o', color='red', label='optimal')
plt.title("Optimal Action Selected at Time-Step", fontsize=18)
plt.xlabel("Time-Step", fontsize=16)
plt.ylabel("Boolean", fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.savefig("actions_plot.pdf", format='pdf', bbox_inches='tight')
plt.close()