import numpy as np

from operator import itemgetter

from toyml.base import Classifier

def _gini(y):
	y = np.array(y)
	p = np.unique(y, return_counts=True)[1] / y.size
	return sum(p * (1 - p))

def _splits(x, y):
	uniques = np.unique(x)
	total_size = y.size
	for unique in uniques[:-1]:
		left_y = y[x <= unique]
		right_y = y[x > unique]
		left_size = left_y.size
		right_size = right_y.size

		yield (unique, 
			   (_gini(left_y) * left_size + 
			   	_gini(right_y) * right_size) / total_size,
			   min(left_size, right_size))

def _intra_best(splits, min_samples=1):
	'''
	Get the best split of an iterable of splits, as defined by the split with the lowest impurity that also has at least `min_samples` on each branch.

	Parameters
	----------

	splits: Iterable
		An iterable of splits of a single feature's space.

	min_samples: int
		The minimum number of samples for a split to be considered viable.
	'''

	try:
		return min(filter(lambda split: split[2] != 0,
					  splits), 
			   key=itemgetter(1))

	except ValueError:
		return None

class Node:

	def __init__(self, X, y):
		self.impurity = _gini(y)

		# Create an interator of (feature_index, split_on, weighted_impurity, minimum_samples) tuples and sort by ascending impurity.

		best_feature_splits = filter(lambda x: x[1] is not None, ((j, *_intra_best(_splits(X[:, j], y))) 
																   for j in range(X.shape[1])))
		try:
			best_split = min(best_feature_splits, key=itemgetter(2))
			if best_split[2] < self.impurity:
				print('yay making split')
				self._split = {'feature_index': best_split[0],
					 		   'on':			best_split[1]}
				self.left = Node(*self.split_lower(X, y))
				self.right = Node(*self.split_upper(X, y))

			else:
				print(self.impurity)
				print(best_split)
				print('impurity is not lower, stopping')

		except:
			print('no candidate splits left?')



	def split_lower(self, X, y):
		index = X[:, self._split['feature_index']] <= self._split['on']
		return (X[index], y[index])

	def split_upper(self, X, y):
		index = X[:, self._split['feature_index']] > self._split['on']
		return (X[index], y[index])

class DTC(Classifier):

	def __init__(self, 
				 max_depth=None):
		pass
		
	def _fit(self, X, y):
		self.base = Node(X, y)

	def _predict_proba(self, X):
		pass

import numpy as np

X = np.random.rand(100, 3)
y = (np.sum(X, axis=1) > 1.5).astype(np.int)