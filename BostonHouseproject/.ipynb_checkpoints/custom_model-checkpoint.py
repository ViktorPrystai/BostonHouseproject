import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class CustomLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        return X @ self.coef_
