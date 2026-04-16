# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic


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
    print(train_df.isna().sum())
    train_df = train_df.dropna(subset=['price_CHF'])
    train_df = pd.get_dummies(train_df, columns=['season'], drop_first=False, dtype=float)
    test_df = pd.get_dummies(test_df, columns=['season'], drop_first=False, dtype=float)


    y_train = train_df['price_CHF'].to_numpy()

    X_train_raw = train_df.drop(columns=['price_CHF'])
    X_test_raw = test_df

    imputer = KNNImputer(n_neighbors=5)
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)


    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    
    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self.kernel = Matern() 
        self.model = GaussianProcessRegressor(kernel=self.kernel, random_state=42)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):

        #TODO: Define the model and fit it using (X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X_test)
        #TODO: Use the model to make predictions y_pred using test data X_test
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

