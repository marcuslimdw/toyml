import numpy as np

from toyml.utils import to_checked_X_y, to_checked_array
from toyml.metrics import accuracy, r_squared


class Estimator:
    '''Base class for estimators.'''

    def _check_fit(self):
        return True

    def fit(self, X, y):
        X_array, y_array = to_checked_X_y(X, y)
        self._fit(X_array, y_array)

    def predict(self, X):
        self._check_fit()
        X_array = to_checked_array(X, 2)
        return self._predict(X_array)


class Classifier(Estimator):
    '''Base class for estimators performing classification.'''

    def _fit(self, X, y):
        return self

    def _predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        return self.labels[np.argmax(self._predict_proba(X), axis=1)]

    def predict_proba(self, X):
        X_array = to_checked_array(X, 2)
        return self._predict_proba(X_array)

    def score(self, X, y_true, metric=accuracy):
        X_array, y_array = to_checked_X_y(X, y_true)
        return metric(self.predict(X_array), y_array)


class Regressor(Estimator):
    '''Base class for estimators performing regression.'''

    def _fit(self, X, y):
        return self

    def _predict(self, X):
        return self

    def score(self, X, y_true, metric=r_squared):
        X_array, y_array = to_checked_X_y(X, y_true)
        return metric(self.predict(X_array), y_array)
