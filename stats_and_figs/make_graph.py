import json
import matplotlib.pyplot as plt
import numpy as np

results_file = "../experiment_results/results_123.json"
ground_truth_file = "../experiment_results/ground_truth_123.json"

def open_json_file(file_name):
    with open(file_name, "r") as json_file:
        data = json.load(json_file)
    return data

data = open_json_file(results_file)
ground_truth = open_json_file(ground_truth_file)

# Extract the episode and reward data
stat_rewards = data["stationary_agent"]["rewards"]
non_stat_rewards = data["naive_non_stat"]["rewards"]
oracle_rewards = data["oracle"]["rewards"]
T = len(stat_rewards)
# Create a line plot
plt.figure(figsize=(8, 6))
plt.plot(range(T), stat_rewards, marker='o', linestyle='-', label='stationary')
plt.plot(range(T), non_stat_rewards, marker='o', linestyle='-', label='non-stationary')
plt.plot(range(T), oracle_rewards, marker='o', linestyle='-', label='oracle')
plt.title("Reward Over Time")
plt.xlabel("Time-Step")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.savefig("reward_plot.pdf")
plt.close()

cummulative_stat_rewards = [np.sum(stat_rewards[:t]) for t in range(T)]
cummulative_non_stat_rewards = [np.sum(non_stat_rewards[:t]) for t in range(T)]
cummulative_oracle_rewards = [np.sum(oracle_rewards[:t]) for t in range(T)]
# cumulative reward across time
plt.figure(figsize=(8, 6))
plt.plot(range(T), cummulative_stat_rewards, marker='o', linestyle='-', label='stationary')
plt.plot(range(T), cummulative_non_stat_rewards, marker='o', linestyle='-', label='non-stationary')
plt.plot(range(T), cummulative_oracle_rewards, marker='o', linestyle='-', label='oracle')
plt.title("Cumulative Reward Over Time")
plt.xlabel("Time-Step")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.savefig("cummulative_reward_plot.pdf")
plt.close()

optimal_actions = ground_truth["optimal action"]
stat_actions = data["stationary_agent"]["actions"]
non_stat_actions = data["naive_non_stat"]["actions"]
oracle_actions = data["oracle"]["actions"]
cummulative_stat_actions = [np.sum([optimal_actions[i] == stat_actions[i] for i in range(t)]) for t in range(T)]
cummulative_non_stat_actions = [np.sum([optimal_actions[i] == non_stat_actions[i] for i in range(t)]) for t in range(T)]
cummulative_oracle_actions = [np.sum([optimal_actions[i] == oracle_actions[i] for i in range(t)]) for t in range(T)]
stat_actions_match = [optimal_actions[i] == stat_actions[i] for i in range(T)]
non_stat_actions_match = [optimal_actions[i] == non_stat_actions[i] for i in range(T)]
oracle_actions_match = [optimal_actions[i] == oracle_actions[i] for i in range(T)]

# comparing to ground truth
plt.figure(figsize=(8, 6))
plt.plot(range(T), cummulative_stat_actions, linestyle='-', label='stationary')
plt.plot(range(T), cummulative_non_stat_actions, linestyle='-', label='non-stationary')
plt.plot(range(T), cummulative_oracle_actions, linestyle='-', label='oracle')
plt.plot(range(1, T + 1), range(1, T + 1), color='red', label='optimal')
plt.title("Total Num. Times Chose Optimal Action Up to Time-Step")
plt.xlabel("Time-Step")
plt.ylabel("Count")
plt.grid(True)
plt.legend()
plt.savefig("cummulative_actions_plot.pdf")
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(range(T), stat_actions_match, marker='o', label='stationary')
plt.scatter(range(T), non_stat_actions_match, marker='o', label='non-stationary')
plt.scatter(range(T), oracle_actions_match, marker='o', label='oracle')
plt.scatter(range(T), np.ones(T), marker='o', color='red', label='optimal')
plt.title("Optimal Action Selected at Time-Step")
plt.xlabel("Time-Step")
plt.ylabel("Boolean")
plt.grid(True)
plt.legend()
plt.savefig("actions_plot.pdf")
plt.close()