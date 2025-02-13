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

def get_total_regret_for_single_trial(actions, ground_truth, k):
    reward_means = actions_to_reward_means(actions, ground_truth)
    optimal_actions = ground_truth["optimal action"]
    optimal_reward_means = actions_to_reward_means(optimal_actions, ground_truth)

    return np.sum(optimal_reward_means[k:] - reward_means[k:])

def get_exp_results_for_single_trial(exp_data, ground_truth):
    k = ground_truth["k"]
    all_total_regrets = []
    for agent_name, _ in exp_data.items():
        actions = exp_data[agent_name]["actions"]
        all_total_regrets.append(get_total_regret_for_single_trial(actions, ground_truth, k))

    assert np.array(all_total_regrets).shape == (len(exp_data.items()),)

    return np.array(all_total_regrets)

def grid_to_latex(matrix, agent_names, title):
    # Start constructing the LaTeX table
    latex_code = f"\\textbf{{{title}}}\\\\\n"  # Adds a bold title above the table
    latex_code += "\\vspace{10pt}\n"  # Adds a little space between the title and the table
    latex_table = "\\begin{tabular}{c" + "c" * (matrix.shape[0] + 1) + "}\n"
    latex_table += "\\hline\n"

    # First row: the assigned numbers for agents
    header = [""] + [str(i + 1) for i in range(matrix.shape[0])] + ["Avg."]
    latex_table += " & ".join(header) + " \\\\\n"
    latex_table += "\\hline\n"

    # Add the matrix data with agent names in the first column
    for i, agent in enumerate(agent_names):
        row = [f"{i + 1}. {agent}"] + [str(matrix[i, j]) if matrix[i, j] != -1 else "-" for j in range(matrix.shape[1])]
        latex_table += " & ".join(row) + " \\\\\n"
    latex_table += "\\hline\n"

    latex_table += "\\end{tabular}\n"
    latex_table += "\\\\\n"
    latex_table += "\\vspace{10pt}\n"

    return latex_code + latex_table

### Note: assumes that the "Kalman Filter agent" is the last idx in agent_names and total_regrets ###
def create_pairwise_table(agent_names, total_regrets, k, sigma_z, sigma_r):
    plt.figure(figsize=(8, 6))
    k = int(k)
    n = len(agent_names[:-1])
    max_seed = len(total_regrets)
    grid = np.zeros((n, n))
    row_averages = np.zeros(n)
    for i in range(n):
        total_sum = 0
        for j in range(n):
            grid[i, j] = np.sum(total_regrets[:, i] < total_regrets[:, j]) / max_seed if i != j else -1
            if i != j:
                total_sum += grid[i, j]
        row_averages[i] = round(total_sum / (n - 1), 2)
    grid = np.hstack((grid, row_averages.reshape(-1, 1)))
    print(grid_to_latex(grid, agent_names[:-1], f"$k = {k}$, $\sigma_z = {sigma_z}$, $\sigma_r = {sigma_r}$"))

def get_all_exp_results(parent_folder):
    subfolder = list_subfolders(parent_folder)
    for exp in subfolder:
        k, sigma_z, sigma_r = extract_values(exp)
        subfolder_path = os.path.join(parent_folder, exp)
        exp_data = open_json_file(subfolder_path + RESULTS_FILE.format(0))
        agents = [key for key, _ in exp_data.items()]
        # ground-truth results appended at the end
        total_regrets = np.zeros((MAX_SEED, len(agents)))
        for exp_seed in range(MAX_SEED):
            exp_data = open_json_file(subfolder_path + RESULTS_FILE.format(exp_seed))
            ground_truth = open_json_file(subfolder_path + GROUND_TRUTH_FILE.format(exp_seed))
            total_regrets[exp_seed] = get_exp_results_for_single_trial(exp_data, ground_truth)
        create_pairwise_table(agents, total_regrets, k, sigma_z, sigma_r)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER + "reward_means/")
    os.makedirs(OUTPUT_FOLDER + "regret/")
get_all_exp_results(EXP_ROOT_FOLDER)