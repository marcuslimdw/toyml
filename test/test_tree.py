import unittest

import numpy as np

from toyml.tree import DecisionTreeClassifier

class TestDecisionTreeClassifier(unittest.TestCase):

	tree = DecisionTreeClassifier(max_depth=10)

	def test_and(self):
		X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
		y = np.array([0, 0, 0, 1])
		self.tree.fit(X, y)
		self.assertTrue((self.tree.predict(X) == y).all())

	def test_or(self):
		X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
		y = np.array([0, 1, 1, 1])
		self.tree.fit(X, y)
		self.assertTrue((self.tree.predict(X) == y).all())

	def test_xor(self):
		X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
		y = np.array([0, 1, 1, 0])
		self.tree.fit(X, y)
		self.assertTrue((self.tree.predict(X) == y).all())

# class TestTree(unittest.TestCase):

	# def test_gini_1(self):
	# 	data = [1, 1, 1, 1, 1]
	# 	expected = 0.0
	# 	self.assertAlmostEqual(_gini(data), expected)

	# def test_gini_2(self):
	# 	data = [1, 2, 3, 4, 5]
	# 	expected = 0.8
	# 	self.assertAlmostEqual(_gini(data), expected)

	# def test_gini_3(self):
	# 	data = [1, 1, 1, 2, 3]
	# 	expected = 0.56
	# 	self.assertAlmostEqual(_gini(data), expected)

	# def test_splits_1(self):
	# 	data_x = np.array([1, 1, 2, 2, 2])
	# 	data_y = np.array([0, 0, 1, 1, 1])
	# 	expected = ((1, 0.0),)
	# 	self.assertTrue(np.isclose(list(_splits(data_x, data_y)),
	# 							   expected).all())

	# def test_splits_2(self):
	# 	data_x = np.array([1, 2, 3, 4, 5])
	# 	data_y = np.array([0, 0, 1, 1, 1])
	# 	expected = np.array([(1, 0.3),
	#  						 (2, 0.0),
	#  						 (3, 4/15),
	#  						 (4, 0.4)])
	# 	self.assertTrue(np.isclose(list(_splits(data_x, data_y)),
	# 							   expected).all())

	# def test_splits_3(self):
	# 	data_x = np.array([1, 1, 1, 2, 3])
	# 	data_y = np.array([0, 0, 1, 1, 1])
	# 	expected = ((1, 4/15),
	# 				(2, 0.4))
	# 	self.assertTrue(np.isclose(list(_splits(data_x, data_y)),
	# 							   expected).all())

unittest.main()