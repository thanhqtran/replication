# estimate parameters eta, phi, theta, eta using MLE
# data is the dataframe data_est
# equations that use the parameters are w_, e_, n_
# y is gdp, n is fertility, e is education, v is productivity

# some libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import r2_score  #to estimate fitness

# import the cleaned dataset
dat = pd.read_csv('dataset.csv')
data = pd.DataFrame(dat)

# helper functions:
# equations to estimate the parameters
# w = y/(1-phi*n)
# e = (eta*phi*w - theta)/(1-eta) if w > theta/(eta*phi) else 0
# n = (1-eta)*gamma*w/((1+gamma)*(phi*w - theta)) if w > theta/(eta*phi) else gamma/(phi*(1+gamma))
# observed variables: y, n, e+theta
# latent variables: w, e
# parameters to estimate: eta, phi, theta, gamma
# assume that residuals are normally distributed

# model's implied values
def compute_model_values(y, n, eta, phi, theta, gamma):
    w = y / (1 - phi * n)
    w_thresh = theta / (eta * phi)

    e = np.where(w > w_thresh, (eta * phi * w - theta) / (1 - eta), 0.0)
    e_plus_theta = e + theta

    n_model = np.where(
        w > w_thresh,
        (1 - eta) * gamma * w / ((1 + gamma) * (phi * w - theta)),
        gamma / (phi * (1 + gamma))
    )

    return w, e_plus_theta, n_model

# log-likelihood function
def neg_log_likelihood(params, y_data, n_data, e_plus_theta_data):
    eta, phi, theta, gamma, log_sigma_n, log_sigma_e = params
    sigma_n = np.exp(log_sigma_n)
    sigma_e = np.exp(log_sigma_e)

    # Boundary check
    if not (0 < eta < 1 and 0 < phi < 1 and theta > 0 and gamma > 0):
        return 1e10  # large penalty for infeasible parameters

    try:
        _, e_plus_theta_model, n_model = compute_model_values(y_data, n_data, eta, phi, theta, gamma)
    except:
        return 1e10  # handle potential division by zero

    # calculate the residual fromm data to model
    res_n = n_data - n_model
    res_e = e_plus_theta_data - e_plus_theta_model
    
    # f(x) = 1/(sqrt{2 \pi \sigma^2} \exp( - x^2/(2\sigma^2))
    # \log f(x) = -1/2 * (\log(2\pi\sigma^2) + x^2/\sigma^2)
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma_n ** 2) + (res_n ** 2) / sigma_n ** 2) \
         -0.5 * np.sum(np.log(2 * np.pi * sigma_e ** 2) + (res_e ** 2) / sigma_e ** 2)
    
    return -ll  # minimize negative log-likelihood

# perform mle
y_data, n_data, e_plus_theta_data = data['y'].values, data['n'].values, data['e+theta'].values
initial_guess = [0.5, 0.03, 60, 0.1, np.log(0.1), np.log(0.1)]
result = minimize(
    neg_log_likelihood, 
    initial_guess, 
    args=(y_data, n_data, e_plus_theta_data), 
    method='L-BFGS-B',
    options={'disp': True}
)
estimated_params = result.x
print("Estimated Parameters:")
print(f"eta: {estimated_params[0]:.4f}, phi: {estimated_params[1]:.4f}, theta: {estimated_params[2]:.4f}, gamma: {estimated_params[3]:.4f}")

# store estimated parameters
estimated_params = result.x
eta, phi, theta, gamma, log_sigma_n, log_sigma_e = estimated_params
# Compute the model-implied latent variables:
w_hat, e_plus_theta_hat, n_hat = compute_model_values(y_data, n_data, eta, phi, theta, gamma)
e_hat = e_plus_theta_hat - theta

# Compute R^2 scores
#r2_score = lambda true, pred: 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
r2_n = r2_score(n_data, n_hat)
r2_e = r2_score(e_plus_theta_data - theta, e_hat)

print(f"R² for n: {r2_n:.4f}")
print(f"R² for e: {r2_e:.4f}")

# Compute Standard Errors, t-Statistics, and p-Values of estimated paramters
from scipy.optimize import approx_fprime
from scipy.linalg import inv
from scipy.stats import norm

# Numerical gradient and Hessian
# calculation of std. dev follows here:
# https://www.sherrytowers.com/mle_introduction.pdf#page=8.81
def grad_loglik(params):
    eps = np.sqrt(np.finfo(float).eps)
    return approx_fprime(params, lambda p: -neg_log_likelihood(p, y_data, n_data, e_plus_theta_data), eps)

def hessian(f, params, epsilon=1e-5):
    n = len(params)
    hess = np.zeros((n, n))
    fx = f(params)
    for i in range(n):
        x1 = np.array(params)
        x1[i] += epsilon
        f1 = f(x1)
        for j in range(i, n):
            x2 = np.array(params)
            x2[j] += epsilon
            if i == j:
                f2 = f(x2)
                hess[i, j] = (f2 - 2 * f1 + fx) / (epsilon ** 2)
            else:
                x3 = np.array(params)
                x3[i] += epsilon
                x3[j] += epsilon
                f3 = f(x3)
                hess[i, j] = hess[j, i] = (f3 - f1 - f1 + fx) / (epsilon ** 2)
    return hess

# Hessian at optimum
hess = hessian(lambda p: -neg_log_likelihood(p, y_data, n_data, e_plus_theta_data), estimated_params)

# Invert to get covariance matrix
cov_matrix = inv(hess)

# Standard errors
standard_errors = np.sqrt(np.diag(cov_matrix))

# t-stats and p-values
t_stats = estimated_params / standard_errors
p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

# Print results
param_names = ['eta', 'phi', 'theta', 'gamma', 'log(sigma_n)', 'log(sigma_e)']
for i, name in enumerate(param_names):
    print(f"{name:<12}: {estimated_params[i]:.4f} ± {standard_errors[i]:.4f}, "
          f"t = {t_stats[i]:.2f}, p = {p_values[i]:.4f}")

