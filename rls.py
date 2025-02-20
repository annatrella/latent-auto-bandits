import numpy as np

def fit_rls(V_t, b_t, x_t, r_t):
    V_t += x_t.reshape(-1, 1) @ x_t.reshape(-1, 1).T
    b_t += r_t * x_t

    return V_t, b_t, np.linalg.inv(V_t) @ b_t

def batch_fit_rls(X, R, lambda_reg):
    V_t = lambda_reg * np.eye(X.shape[1])
    b_t = np.zeros(X.shape[1])
    theta_t = np.linalg.inv(V_t) @ b_t

    for x_t, r_t in zip(X, R):
        V_t, b_t, theta_t = fit_rls(V_t, b_t, x_t, r_t)

    return V_t, b_t, theta_t