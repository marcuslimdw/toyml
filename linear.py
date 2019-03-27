import numpy as np

from toyml.base import Regressor
from toyml.metrics import r_squared


class LinearRegression(Regressor):

    __slots__ = ['fit_intercept', 'epochs', 'learning_rate', 'method']

    def __init__(self, fit_intercept=True, epochs=1000, learning_rate=0.05, method='simple'):
        self.fit_intercept = fit_intercept
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.method = method

    def _loss(self, X, y):
        return (self.predict(X) - y) ** 2

    def _fit(self, X, y):
        if self.fit_intercept:
            X_padded = np.concatenate([X, np.ones((len(X), 1))], axis=1)
            initial_weights = np.zeros(X.shape[1] + 1)

        else:
            X_padded = np.asarray(X)
            initial_weights = np.zeros(X.shape[1])

        n_splits = round(X.shape[0] ** 0.5)
        self.weights = self._descend(X_padded, y, initial_weights, self.epochs, n_splits, self.learning_rate)

    def _descend(self, X, y, w, epochs, n_splits, learning_rate):
        def iterate(X_sub, y_sub, learning_rate):
            y_hat = (X_sub * w).sum(axis=1)
            errors = y_sub - y_hat
            error_sums = (errors * X_sub.T).sum(axis=1)
            correction = learning_rate * error_sums / n_rows
            return w + correction

        n_rows = X.shape[0]
        for i in range(epochs):
            random_indices = np.random.permutation(n_rows)
            splits = np.array_split(random_indices, n_splits)
            for split in splits:
                X_sub = X[split]
                y_sub = y[split]
                w = iterate(X_sub, y_sub, learning_rate)

        return w

    def _predict(self, X):
        return np.sum(X * self.weights, axis=1)

    def score(self, X, y_true, metric=r_squared):
        if self.fit_intercept:
            X_padded = np.concatenate([X, np.ones((len(X), 1))], axis=1)

        else:
            X_padded = X

        return super().score(X_padded, y_true, metric)
