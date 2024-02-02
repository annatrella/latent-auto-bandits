import numpy as np
from simulations import *

def generate_centered_stable_weights(order, range_factor, seed):
    np.random.seed(seed)
    # Generate random weights within a specified range
    weights = np.random.uniform(low=-1 * range_factor, high=range_factor, size=order)
    
    # Ensure centering by adjusting the weights
    weights = weights - np.mean(weights)
    weights = weights / np.sum(np.abs(weights))
    weights = np.array([ 0.34175598,  1.87617084,  0.95042384, -0.57690366, -0.89841467])
    
    # Ensure stability by checking if the roots are inside the unit circle
    while np.max(np.abs(np.roots([1] + list(-weights)))) >= 1:
        weights = np.random.uniform(low=-1 * range_factor, high=range_factor, size=order)
        weights = weights - np.mean(weights)
        weights = weights / np.sum(np.abs(weights))

    return weights

### TRUE ENV. PARAMS ###
# K = 5
# NUM_TIME_STEPS = 100
# K = 1
# gammas = [0.9]
# INIT_ZS = [1]
# gammas = generate_centered_stable_weights(K, 2, 123)
# print("GAMMAS", gammas)  
# assert np.sum(gammas) < 1

# INIT_ZS = np.ones(K)
# no_noise_env_params = {
#     "K": K,
#     "noise_std": 0,
#     "gamma_0": 0,
#     "gammas": gammas,
#     "beta_0": [0, 0],
#     "beta_1": [-1.0, 1.0],
#     "init_zs": INIT_ZS
#     # "init_zs": np.zeros(K)
# }
# low_noise_env_params = {
#     "K": K,
#     "noise_std": 1,
#     "gamma_0": 0,
#     "gammas": gammas,
#     "beta_0": [0, 0],
#     "beta_1": [-1.0, 1.0],
#     "init_zs": INIT_ZS
# }
# high_noise_env_params = {
#     "K": K,
#     "noise_std": 5,
#     "gamma_0": 0,
#     "gammas": gammas,
#     "beta_0": [0, 0],
#     "beta_1": [-1.0, 1.0],
#     "init_zs": INIT_ZS
# }

# def calculate_ground_truth(env, seed):
#     np.random.seed(seed)
#     T = env.get_T()
#     K = env.K
#     ground_truth = {
#         "zs": np.empty(T)
#     }
#     for _ in range(K):
#         t = env.get_t()
#         ground_truth["zs"][t] = env.get_all_zs()[t]  
#         env.increment_t()   
#     while env.get_t() < T:
#         ### environment ###
#         t = env.get_t()
#         # print(f"Time Step: {t}")
#         env.state_evolution()
#         ground_truth["zs"][t] = env.get_all_zs()[t]
#         ### increment t ###  
#         env.increment_t()

#     return ground_truth

# zs_high_noise = calculate_ground_truth(Environment(high_noise_env_params, T=NUM_TIME_STEPS), 1)
# zs_low_noise = calculate_ground_truth(Environment(low_noise_env_params, T=NUM_TIME_STEPS), 1)
# zs_no_noise = calculate_ground_truth(Environment(no_noise_env_params, T=NUM_TIME_STEPS), 1)

# import matplotlib.pyplot as plt

# def plot_zs(zs, fig_name, k):
#     plt.figure(figsize=(8, 6))
#     plt.plot(range(NUM_TIME_STEPS), zs["zs"])
#     plt.xlabel("Time-Step", fontsize=16)
#     plt.ylabel("z", fontsize=16)
#     plt.title(f"$k = {k}$")
#     plt.grid(True)
#     plt.savefig(f"{fig_name}.pdf", format='pdf', bbox_inches='tight')
#     plt.close()

# plot_zs(zs_high_noise, 'zs_high_noise', K)
# plot_zs(zs_low_noise, 'zs_low_noise', K)
# plot_zs(zs_no_noise, 'zs_no_noise', K)