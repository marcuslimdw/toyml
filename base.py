from toyml.utils import check_array_like

class Estimator:
	'''Base class for estimators.'''

	def _check_fit(self):
		return True
	
	def fit(self, X, y):
		X, y = check_array_like(X, y)
		
		self._fit(X, y)

	def predict(self, X):
		self._check_fit()
		self._predict(X)

	def score(self, X, y):
		self._check_fit()
		self._score(y, self._predict(X))

class Classifier(Estimator):
	'''Base class for estimators performing classification.'''

	def _fit(self, X, y):
		pass

	def _predict_proba(self, X):
		pass

	def _predict(self, X):
		return np.argmax(self._predict_proba(X), 
						 axis=1)

	def _score(self, y_true, y_pred):
		pass

	def predict_proba(self, X):
		return self._predict_proba(X)

class Regressor(Estimator):
	'''Base class for estimators performing regression.'''

	def _fit(self, X, y):
		pass

	def _predict(self, X):
		pass

	def _score(self, y_true, y_pred):
		pass