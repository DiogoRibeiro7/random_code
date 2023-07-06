import numpy as np

def coxph_fit(X, T, E, alpha=0.05, max_iter=100):
    n = X.shape[0]  # Number of samples
    p = X.shape[1]  # Number of features
    
    # Initialize coefficients
    beta = np.zeros(p)
    
    for _ in range(max_iter):
        # Compute the baseline hazard
        hazards = np.exp(np.dot(X, beta))
        baseline_hazard = np.sum(hazards[E == 1]) / np.sum(E == 1)
        
        # Compute the partial likelihood
        log_likelihood = 0.0
        risk_scores = np.dot(X, beta)
        for i in range(n):
            exp_sum = np.sum(np.exp(risk_scores[E == 1]))
            log_likelihood += risk_scores[i] - np.log(exp_sum)
            
        # Compute the score vector and Hessian matrix
        score = np.zeros(p)
        hessian = np.zeros((p, p))
        for j in range(p):
            score[j] = np.sum(X[:, j] * (E - hazards / baseline_hazard))
            for k in range(p):
                hessian[j, k] = -np.sum(X[:, j] * X[:, k] * hazards / baseline_hazard)
        
        # Perform Newton-Raphson update
        beta_new = beta - np.linalg.inv(hessian) @ score
        if np.linalg.norm(beta_new - beta) < alpha:
            break
        beta = beta_new
    
    return beta

# Example usage
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
T = np.array([5, 7, 10, 8])
E = np.array([1, 1, 0, 1])

beta_estimated = coxph_fit(X, T, E)
print("Estimated coefficients:", beta_estimated)
