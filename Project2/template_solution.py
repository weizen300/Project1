# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def handleNans(X: np.ndarray, method: str) -> np.ndarray:
    if method == 'mean':
        colMeans = np.nanmean(X, axis=0)

        nan_rows, nan_cols = np.where(np.isnan(X))

        X[nan_rows, nan_cols] = colMeans[nan_cols]
    if method == 'median':
        colMedians = np.nanmedian(X, axis=0)

        nan_rows, nan_cols = np.where(np.isnan(X))

        X[nan_rows, nan_cols] = colMedians[nan_cols]
    if method == 'iterative':
        imputer = IterativeImputer()
        X = imputer.fit_transform(X)
    return X


def load_data():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables
    y_train = train_df['price_CHF'].to_numpy()
    X_train = pd.get_dummies(
        train_df.drop(['price_CHF'],axis=1),
        columns=['season'],
        dtype='float'
    ).to_numpy()
    X_test = pd.get_dummies(
        test_df,
        columns=['season'],
        dtype='float'
    ).to_numpy()

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    indices = np.where(~np.isnan(y_train))[0]
    y_train = y_train[indices]
    X_train = X_train[indices, :]

    # Fill the NaNs
    X_train = handleNans(X_train, method='iterative')
    X_test = handleNans(X_test, method='iterative')

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None
        self.model = GaussianProcessRegressor(kernel=Matern(), alpha=1e-1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        #TODO: Define the model and fit it using (X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X_test)
        #TODO: Use the model to make predictions y_pred using test data X_test
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# =============================================================================
# PLAYGROUND / TESTING ONLY — not part of the solution
# =============================================================================
from sklearn.model_selection import cross_val_score

def evaluate_model(X_train, y_train, kernel, alpha=1e-10, cv=10):
    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    print(f"Kernel: {kernel}")
    print(f"R² scores per fold: {scores}")
    print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")
    print("=====================================================")
    return scores.mean()
# =============================================================================
# END PLAYGROUND
# =============================================================================


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = load_data()
    model = Model()
    # Use this function to fit the model
    model.fit(X_train=X_train, y_train=y_train)
    # Use this function for inference
    y_pred = model.predict(X_test)

    # best_idx = -1, -1
    # best_value = -1.
    #
    # kernels = ["RBF", "DotProduct", "Matern", "RationalQuadratic"]
    # alphas = [100, 10., 5., 1., 1e-1, 1e-2]
    # for i, kernel in enumerate([RBF(), DotProduct(), Matern(), RationalQuadratic()]):
    #     for j, alpha in enumerate([100, 10., 5., 1., 1e-1, 1e-2]):
    #         current_value = evaluate_model(X_train, y_train, kernel, alpha, cv=5)
    #         if  current_value > best_value:
    #             best_value = current_value
    #             best_idx = i, j
    # print(f"best kernel, alpha: {kernels[best_idx[0]]}, {alphas[best_idx[1]]}")
    # print(f"best value: {best_value}")

    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

