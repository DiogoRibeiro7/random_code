import numpy as np

def soft_threshold(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

def gfl(y, lambda_, tol=1e-8, max_iter=500):
    n = len(y)
    beta = np.zeros(n)
    for _ in range(max_iter):
        beta_old = beta.copy()
        gradient = -2 * (y - beta)
        beta_unthresholded = beta - (1.0 / (2 * 1.0)) * gradient
        for i in range(1, n):
            beta[i] = soft_threshold(beta_unthresholded[i], lambda_)
        if np.sum((beta - beta_old)**2) < tol:
            break
    return beta

# Test the function with some data
np.random.seed(0)
n = 100
y = np.concatenate([np.ones(n), -np.ones(n), np.ones(n)]) + 0.1 * np.random.randn(3*n)
lambda_ = 0.5
beta_hat = gfl(y, lambda_)
