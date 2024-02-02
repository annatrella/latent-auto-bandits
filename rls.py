import numpy as np

def fit_rls(V_t, b_t, x_t, r_t):
    V_t += x_t.reshape(-1, 1) @ x_t.reshape(-1, 1).T
    b_t += r_t * x_t

    return V_t, b_t, np.linalg.inv(V_t) @ b_t