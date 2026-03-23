# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Add any additional imports here (however, the task is solvable without using
# any additional imports)
# import ...

def transform_features(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component in a given row of X)
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: matrix of floats: dim = (700,21), transformed input with 21 features
    """
    # TODO: Enter your code here
    X_linear = X
    X_quadratic = X**2
    X_exponential = np.exp(X)
    X_cosine = np.cos(X)
    X_costant = np.ones((700, 1))

    X_transformed = np.hstack((X_linear, X_quadratic, X_exponential, X_cosine, X_costant))
    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit_logistic_regression(X, y, lam=0.):
    """
    This function receives training data points, transforms them, and then fits the logistic regression on this 
    transformed data. Finally, it outputs the weights of the fitted logistic regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of integers in {0,1}, dim = (700,), input labels

    Returns
    ----------
    weights: array of floats: dim = (21,), optimal parameters of logistic regression
    """
    n_samples = X.shape[0]
    X_transformed = transform_features(X)
    weights = np.zeros((21,))

    # --- Adam Hyperparameters ---
    alpha = 0.01  # Step size (learning rate)
    beta1 = 0.9  # Exponential decay rate for the first moment
    beta2 = 0.999  # Exponential decay rate for the second moment
    epsilon = 1e-8  # To prevent division by zero

    # Initialize moments
    m = np.zeros(21)
    v = np.zeros(21)

    epochs = 150_000  # Adam converges much faster than basic GD

    for t in range(1, epochs + 1):
        # 1. Calculate Gradient (standard Logistic Regression gradient)
        # Using a stable sigmoid helper
        z = np.clip(X_transformed @ weights, -250, 250)
        h = 1. / (1. + np.exp(-z))

        # Regularization part (don't penalize bias)
        w_reg = weights.copy()
        w_reg[-1] = 0
        gradient = (1. / n_samples) * (X_transformed.T @ (h - y)) + 2 * lam * w_reg

        # 2. Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * gradient

        # 3. Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (gradient ** 2)

        # 4. Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)

        # 5. Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)

        # 6. Update Weights
        weights = weights - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        if t % 10000 == 0: print(f"Iteration {t}")
    return weights

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit_logistic_regression(X, y, 0)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")

    lambdas = [0]
    f1_scores = []
    for lam in lambdas:
        w_lam = fit_logistic_regression(X, y, lam)
        X_t = transform_features(X)
        probs = 1. / (1 + np.exp(-X_t @ w_lam))
        y_pred = (probs >= 0.5).astype(int)
        TP = np.sum((y_pred == 1) & (y == 1))
        FP = np.sum((y_pred == 1) & (y == 0))
        FN = np.sum((y_pred == 0) & (y == 1))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        print(f"  lam={lam:.2e}  F1={f1:.4f}")

    best_idx = 0
    print(f"\nBest lambda: {lambdas[best_idx]:.2e}  ->  F1={f1_scores[best_idx]:.16f}")