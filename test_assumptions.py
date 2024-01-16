import numpy as np

gamma_0 = 0.2
gammas = np.array([0.2, 0.1, 0.3, 0.2, 0.1])
beta_0 = [0.25, -0.25]
beta_1 = [1.0, 1.5]

def calculate_diff_forward(gamma_0, gammas, beta_0, beta_1):
    return (beta_0[0] - beta_0[1])*(1 + np.sum(gammas)) + gamma_0 * (beta_1[0] - beta_1[1])

def calculate_diff_backward(gamma_0, gammas, beta_0, beta_1):
    return (beta_0[1] - beta_0[0])*(1 + np.sum(gammas)) + gamma_0 * (beta_1[1] - beta_1[0])

print("Satifies Assumption?: ", calculate_diff_forward(gamma_0, gammas, beta_0, beta_1) <= 1)
print("Satifies Assumption?: ", calculate_diff_backward(gamma_0, gammas, beta_0, beta_1) <= 1)