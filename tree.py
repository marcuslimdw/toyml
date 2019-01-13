import numpy as np

from operator import itemgetter

from toyml.base import Classifier

from functional import flatten

def _gini(y):
	'''Calculate the Gini impurity of a dataset of labels. 

	Randomly choose one element from the dataset and assign to it a random label, based on the distribution of labels in the dataset. The probability that this random label is wrong is the Gini impurity.

	Parameters
	----------

	y: 1D iterable
		The dataset to calculate the Gini impurity of.

	Returns
	----------

	Gini: float
		The calculated Gini impurity of `y`.

	Examples
	----------

	Given `y = [1, 1, 1, 1, 1]`, a randomly chosen element can only be 1. Randomly choosing a label from the dataset's distribution of labels can also only give 1. The probability that the two do not match, and therefore the Gini impurity, is 0.

	Given `y = [1, 1, 2, 2, 3]`, the probabilities of choosing each unique label are as follows:

	1: 0.4
	2: 0.4
	3: 0.2

	In the case of a randomly chosen label of 1, there is a 1.0 - 0.6 = 0.4 probability that the correct label is 2 or 3, for a total probability of 0.6 * 0.4 = 0.24 of making a mistake. This similarly applies for class 2. For class 3, the probability of a mistake is 0.2 * 0.8 = 0.16. Summing all of these probabilities up, the Gini impurity of the dataset is 0.64.
	'''

	p = np.unique(y, return_counts=True)[1] / y.size
	return sum(p * (1 - p))

class Node:

	def _weighted_quality(self, left_y, right_y):
		'''Calculate the weighted quality of a split by applying the quality function to each sub-dataset and taking the mean weighted by the size the sub-datasets.'''
		left_size = left_y.size
		right_size = right_y.size
		total_size = left_size + right_size
		return (self._head._quality_func(left_y) * left_size + 
			    self._head._quality_func(right_y) * right_size) / total_size

	def _splits(self, X, y, j):
		x = X[:, j]
		def split_gen(x, y, j):
			uniques = np.unique(x)
			total_size = y.size
			for unique in uniques[:-1]:
				left_y = y[x <= unique]
				right_y = y[x > unique]

				# Splits could also have been implemented as instances of `tuple` or `namedtuple`, which would most likely have been more efficient. I could test that sometime, but this is ultimately a design choice in line with the objective of `toyml`: to write machine learning library emphasising, in general, readability over performance (but it should still be as fast as possible otherwise).

				yield {'feature_index': 	j,
					   'on': 				unique, 
					   'weighted_quality':	self._weighted_quality(left_y, right_y),
					   'n_samples': 		(left_y.size, right_y.size)}

		return tuple(split_gen(x, y, j))

	def _get_best_split(self, X, y):
		def is_binary(split):
			return min(split['n_samples']) > 0

		candidates = (filter(is_binary, self._splits(X, y, j)) for j in range(X.shape[1]))
		try:
			best_split = min(flatten(candidates), key=itemgetter('weighted_quality'))
			
		except ValueError:
			best_split = None

		return best_split

	def __init__(self, head, X, y, depth):
		self._head = head
		self._quality = self._head._quality_func(y)
		self._split = self._get_best_split(X, y)
		self._depth = depth

		if self._terminate():
			value_counts = np.vstack(np.unique(y, return_counts=True)).T
			self._probabilities = {value: count / y.size for value, count in value_counts}
			self._leaf = True

		else:
			self._left = Node(self._head, *self._split_left(X, y), depth=self._depth + 1)
			self._right = Node(self._head, *self._split_right(X, y), depth=self._depth + 1)
			self._leaf = False

	def _terminate(self):
		return any((self._split is None,
					self._depth > self._head.max_depth))

	def _split_left(self, X, y):
		index = X[:, self._split['feature_index']] <= self._split['on']
		return (X[index], y[index])

	def _split_right(self, X, y):
		index = X[:, self._split['feature_index']] > self._split['on']
		return (X[index], y[index])

	def _get_probabilities(self, x):
		if self._leaf:
			return self._probabilities

		else:
			if x[self._split['feature_index']] <= self._split['on']:
				return self._left._get_probabilities(x)

			else:
				return self._right._get_probabilities(x)

	def predict_proba(self, X):
		return tuple(self._get_probabilities(x) for x in X)

class DecisionTreeClassifier(Classifier):

	def __init__(self, 
				 quality_func=_gini,
				 max_depth=3):
		self._quality_func = quality_func
		self.max_depth = max_depth
		
	def _fit(self, X, y):
		self.classes = np.unique(y)
		self.base = Node(self, X, y, 0)

	def _predict_proba(self, X):
		return self.base.predict_proba(X)

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([0, 1, 1, 0])

tree = DecisionTreeClassifier(max_depth=10)
tree._fit(X, y)

# X = np.random.rand(1000, 3)
# y = (X.sum(axis=1) > 1.5).astype(np.int)

# train_indices = np.random.choice(range(1000), size=700)
# test_indices = np.arange(1000)[~np.isin(np.arange(1000), train_indices)]

# X_train = X[train_indices]
# X_test = X[test_indices]
# y_train = y[train_indices]
# y_test = y[test_indices]

# c = DecisionTreeClassifier(max_depth=10)
# c._fit(X_train, y_train)

# y_pred = np.array(tuple(zip(*[max(pair.items(), key=itemgetter(1)) for pair in c._predict_proba(X_test)]))[0])