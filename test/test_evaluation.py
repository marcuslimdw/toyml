import numpy as np

from toyml.evaluation import *

import pytest

dataset_proba = [[np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])],

				 [np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])],

				 [np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])],

				 [np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1]), np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]]

dataset_pred = [(y_true, (y_proba >= 0.5).astype(np.int8)) for (y_true, y_proba) in dataset_proba]

roc_expected = [[np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
				 np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]), 
				 np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])],

				 # I think that this result could, in some way, be considered a bug. It's a necessary consequence of the
				 # algorithm, but it is inefficient.

				[np.array([1.0, 1.0, 1.0, 0.0]), 
				 np.array([1.0, 1.0, 1.0, 0.0]), 
				 np.array([0.0, 0.8, 0.8, 1.0])],

				[np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]), 
				 np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]), 
				 np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])],

				[np.array([1.0, 0.8, 0.8, 0.6, 0.4, 0.4, 0.2, 0.2, 0.2, 0.0, 0.0]), 
				 np.array([1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2, 0.0]), 
				 np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]]

precision_expected = [1.0, 
					  0.5,
					  1.0,
					  0.6]

recall_expected = [1.0, 
				   1.0,
				   0.5,
				   0.6]


@pytest.mark.parametrize('data, expected', zip(dataset_proba, roc_expected))
def test_roc_curve(data, expected):
	if len(np.unique(data[0])) > 1:
		result = roc_curve(*data)
		np.testing.assert_allclose(result, expected)

	else:
		with pytest.raises(ValueError):
			result = roc_curve(*data)


@pytest.mark.parametrize('data, expected', zip(dataset_pred, precision_expected))
def test_precision(data, expected):
	if 1 in data[1]:
		result = precision(*data)
		np.testing.assert_allclose(result, expected)

	else:
		with pytest.warns(RuntimeWarning):
			assert precision(*data) == 0


@pytest.mark.parametrize('data, expected', zip(dataset_pred, recall_expected))
def test_recall(data, expected):
	result = recall(*data)
	np.testing.assert_allclose(result, expected)


# @pytest.mark.parametrize('data, expected', zip(dataset_pred, roc_auc_score_expected))
# def test_roc_auc_score():
# 	result = roc_auc_score(y_true, y_proba)
# 	expected = 0.68
# 	np.testing.assert_allclose(result, expected)


# def test_random_splitter():

# 	# To-do: add more cases
# 	#
# 	#	-	number of rows not evenly divisible
# 	#	-	non-random sampling
# 	#	-	stratified sampling 

# 	splitter = Splitter(6, 3, 1)
# 	y = np.arange(10)
# 	X = np.reshape(y, (10, 1))
# 	(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = splitter.split(X, y, shuffle=True, random_state=0)
# 	np.testing.assert_array_equal(np.reshape(X_train, (X_train.size)), y_train)
# 	np.testing.assert_array_equal(np.reshape(X_valid, (X_valid.size)), y_valid)
# 	np.testing.assert_array_equal(np.reshape(X_test, (X_test.size)), y_test)

@pytest.mark.parametrize('x, y, expected', [((0, 1), (0, 0), 0),
								  			((0, 0, 1, 1), (0, 1, 1, 0), 1),
								  			((0, 1, 1), (0, 1, 0), 0.5),
								  			((0, 0, 1, 1), (0, -1, -1, 0), -1),
								  			((0.0, 0, 0.5, 0.5, 1.0, 1.0), (0.0, 1.0, 1.0, -1.0, -1.0, 0), 0),
								  			((0.0, 0.0, 0.5, 1.0, 1.0), (0.0, 1.0, 1.0, -1.0, 0.0), 0.5),
								  			(np.linspace(0, 2 * np.pi, 200), np.sin(np.linspace(0, 2 * np.pi, 200)), 0),
								  			(np.linspace(0, 2 * np.pi, 200), np.cos(np.linspace(0, 2 * np.pi, 200)), 0),
								  			(np.linspace(0, 1, 200), np.exp(np.linspace(0, 1, 200)), np.e - 1)])
def test_auc(x, y, expected):
	np.testing.assert_allclose(auc(x, y), expected, atol=1e-05)