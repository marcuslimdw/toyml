import numpy as np

from toyml.utils import check_shape

from toyml.evaluation import accuracy, mean_squared_error

class Estimator:
	'''Base class for estimators.'''

	def _check_fit(self):
		return True
	
	def fit(self, X, y):
		check_shape(X, 2)
		check_shape(y, 1)
		self._fit(X, y)

	def predict(self, X):
		self._check_fit()
		return self._predict(X)

class Classifier(Estimator):
	'''Base class for estimators performing classification.'''

	def _fit(self, X, y):
		return self

	def _predict_proba(self, X):
		return self

	def _predict(self, X):
		return np.argmax(self._predict_proba(X), 
						 axis=1)

	def score(self, X, y, metric=accuracy):
		return accuracy(y, self.predict(X))

	def predict_proba(self, X):
		return self._predict_proba(X)

class Regressor(Estimator):
	'''Base class for estimators performing regression.'''

	def _fit(self, X, y):
		return self

	def _predict(self, X):
		return self

	def score(self, X, y, metric=mean_squared_error):
		return accuracy(y, self.predict(X))