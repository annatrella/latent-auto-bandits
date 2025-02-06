import numpy as np
import control as ctrl

def compute_c_a(k, beta_a):
    c_a = np.zeros(k)
    c_a[0] = beta_a
    return c_a

def compute_C(k):
    C = np.zeros(k)
    C[0] = 1
    return C.reshape(1, k)

def create_Gamma_from_gammas(gammas):
    k = len(gammas)
    bases = np.eye(k).tolist()
    Gamma = [gammas]
    for j in range(k - 1):
        Gamma.append(bases[j])
    return np.array(Gamma).reshape(k, k)

# ref: steady-state Kalman Filter 
# https://laurentlessard.com/teaching/me7247/lectures/lecture%2012%20-%20steady-state%20Kalman%20filter.pdf
def compute_K(P, C, V):
    return P @ C.T @ np.linalg.inv(C @ P @ C.T + V)

# ref: https://python-control.readthedocs.io/en/latest/generated/control.dare.html
def compute_P(Gamma, C, W, V):
    A = Gamma.T
    B = C.T
    X, _, _ = ctrl.dare(A, B, W, V)

    return X

def compute_z_tilde(Gamma, K, C, last_z_tilde, mu_z, last_y):
    return (Gamma - Gamma @ K @ C) @ last_z_tilde + mu_z + Gamma @ K @ last_y

# NOTE: beta_as is env_params["beta_a"] and is a list of beta parameters one for each action
def create_lds_from_ar_process(gammas, gamma_0, k, sigma_z, sigma_r, beta_as):
    Gamma = create_Gamma_from_gammas(gammas)
    C = compute_C(k)
    W = np.diag(np.zeros(k))
    W[0][0] = sigma_z**2
    V = np.array([[sigma_r**2 / np.min(np.power(beta_as, 2))]])
    mu_z = np.zeros(k)
    mu_z[0] = gamma_0

    return Gamma, C, W, V, mu_z