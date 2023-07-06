from scipy.stats import chi2
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
                hessian[j, k] = - \
                    np.sum(X[:, j] * X[:, k] * hazards / baseline_hazard)

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


def coxph_fit_with_variable_selection(X, T, E, alpha=0.05, max_iter=100, selection_criterion='wald'):
    n = X.shape[0]  # Number of samples
    p = X.shape[1]  # Number of features

    # Initialize coefficients
    beta = np.zeros(p)

    # Variable selection
    selected_features = set()

    for _ in range(max_iter):
        # Compute the baseline hazard
        hazards = np.exp(
            np.dot(X[:, list(selected_features)], beta[list(selected_features)]))
        baseline_hazard = np.sum(hazards[E == 1]) / np.sum(E == 1)

        # Compute the partial likelihood
        log_likelihood = 0.0
        risk_scores = np.dot(X[:, list(selected_features)],
                             beta[list(selected_features)])
        for i in range(n):
            exp_sum = np.sum(np.exp(risk_scores[E == 1]))
            log_likelihood += risk_scores[i] - np.log(exp_sum)

        # Compute the score vector and Hessian matrix
        score = np.zeros(p)
        hessian = np.zeros((p, p))
        for j in range(p):
            score[j] = np.sum(X[:, j] * (E - hazards / baseline_hazard))
            for k in range(p):
                hessian[j, k] = - \
                    np.sum(X[:, j] * X[:, k] * hazards / baseline_hazard)

        # Perform Newton-Raphson update for selected features
        beta_new = beta.copy()
        beta_new[list(selected_features)] -= np.linalg.inv(hessian[list(selected_features)]
                                                           [:, list(selected_features)]) @ score[list(selected_features)]

        # Compute criterion for all remaining features
        if selection_criterion == 'wald':
            criterion_values = np.zeros(p)
            for j in range(p):
                if j not in selected_features:
                    se_j = np.sqrt(np.linalg.inv(hessian)[j, j])
                    wald_stat = (beta_new[j] / se_j) ** 2
                    criterion_values[j] = wald_stat
                else:
                    criterion_values[j] = -np.inf

        # Select next feature based on the criterion
        best_feature = np.argmax(criterion_values)
        wald_threshold = chi2.ppf(1 - alpha, 1)
        if criterion_values[best_feature] < wald_threshold:
            break
        selected_features.add(best_feature)

        beta = beta_new

    return beta


# Example usage
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
T = np.array([5, 7, 10, 8])
E = np.array([1, 1, 0, 1])

beta_estimated = coxph_fit_with_variable_selection(X, T, E)
print("Estimated coefficients:", beta_estimated)
