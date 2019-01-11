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

		yield (unique, 
			   (_gini(left_y) * left_y.size + 
			   	_gini(right_y) * right_y.size) / total_size)

def _best(splits):
	return min(splits, 
			   key=itemgetter(1))

class Node:

	def __init__(self, feature, split_on):
		pass

class DecisionTreeClassifier(Classifier):

	def __init__(self):
		pass

	def _fit(self, X, y):
		base_impurity = _gini(y)

	def _predict_proba(self, X):
		pass