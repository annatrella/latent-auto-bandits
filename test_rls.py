import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from rls import fit_rls

# Function to simulate toy data
def generate_toy_data(n_samples=100, n_features=5, sigma_z=0.1):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    true_coefficients = np.random.randn(n_features)
    y = X.dot(true_coefficients) + np.random.normal(0, sigma_z, n_samples)
    return X, y, true_coefficients

# Function to fit Ridge regression model
def fit_ridge_regression(X_train, y_train, alpha=1.0):
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    return ridge_model

def fit_my_batch_rls(X_train, y_train, alpha=1.0):
    d = X_train.shape[1]
    return np.linalg.inv(X_train.T @ X_train + alpha * np.eye(d)) @ X_train.T @ y_train

def fit_my_rls(X_train, y_train, alpha=1.0):
    d = X_train.shape[1]
    V_t = alpha * np.eye(d)
    b_t = np.zeros(d)
    theta_t = np.zeros(d)
    for t in range(len(X_train)):
        V_t, b_t, theta_t = fit_rls(V_t, b_t, X_train[t], y_train[t])

    return np.linalg.inv(V_t) @ b_t

# Function to verify learned parameters
def verify_parameters(true_coefficients, learned_coefficients, tolerance=0.2):
    diff = np.abs(true_coefficients - learned_coefficients)
    return np.all(diff < tolerance)

##### Main simulation #####
# X, y, true_coefficients = generate_toy_data(1000)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Fit Ridge regression model
# ridge_model = fit_ridge_regression(X_train, y_train)
# my_rls = fit_my_rls(X_train, y_train, alpha=0.01)

# # Verify learned parameters
# learned_coefficients = ridge_model.coef_
# is_parameters_good = verify_parameters(true_coefficients, learned_coefficients)
# are_my_params_good = verify_parameters(true_coefficients, my_rls)

# # Print results
# print("True Coefficients:", true_coefficients)
# print("Learned Coefficients:", learned_coefficients)
# print("Anna RLS", my_rls)
# print("Are learned parameters good?", is_parameters_good)
# print("Are my manual parameters good?", are_my_params_good)

# # Optionally, evaluate model performance
# y_pred = ridge_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error on Test Set:", mse)

# # my own prediction
# y_pred = X_test @ my_rls
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error on Test Set:", mse)