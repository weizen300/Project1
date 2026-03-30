# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


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

    # One-hot encode the 'season' column (drop_first avoids perfect multicollinearity)
    train_df = pd.get_dummies(train_df, columns=['season'], drop_first=False)
    test_df  = pd.get_dummies(test_df,  columns=['season'], drop_first=False)

    # Align test columns to train (in case a season is missing in test split)
    season_cols = [c for c in train_df.columns if c.startswith('season_')]
    for col in season_cols:
        if col not in test_df.columns:
            test_df[col] = 0

    # Drop rows where the target is NaN — we cannot train on those
    train_df = train_df.dropna(subset=['price_CHF'])

    # Separate target
    y_train = train_df['price_CHF'].to_numpy(dtype=float)

    # Feature columns: everything except the target
    feature_cols = [c for c in train_df.columns if c != 'price_CHF']
    X_train_raw = train_df[feature_cols].to_numpy(dtype=float)
    X_test_raw  = test_df[feature_cols].to_numpy(dtype=float)

    # Impute missing feature values using KNN (k=5).
    # Fit only on training data to avoid leaking test information.
    imputer = KNNImputer(n_neighbors=5)
    X_train = imputer.fit_transform(X_train_raw)
    X_test  = imputer.transform(X_test_raw)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        # RidgeCV automatically selects the best regularisation strength via
        # leave-one-out cross-validation, so no manual tuning is needed.
        self._scaler = StandardScaler()
        self._model  = RidgeCV(alphas=np.logspace(-3, 4, 50), cv=5)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Standardise features (Ridge is sensitive to scale)
        X_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_scaled, y_train)
        print(f"Best Ridge alpha: {self._model.alpha_:.4f}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X_test)
        y_pred = self._model.predict(X_scaled)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = load_data()
    model = Model()
    # Use this function to fit the model
    model.fit(X_train=X_train, y_train=y_train)
    # Use this function for inference
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
