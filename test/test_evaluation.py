import numpy as np

from toyml.evaluation import *

import pytest

# (np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), 
				  # np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])),

dataset_proba = [(np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1]), 
				  np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))]

dataset_pred = [(y_true, (y_proba >= 0.5).astype(np.int8)) for (y_true, y_proba) in dataset_proba]

roc_expected = [np.array([1.0, 0.8, 0.8, 0.6, 0.4, 0.4, 0.2, 0.2, 0.2, 0.0, 0.0]), 
				np.array([1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2, 0.0]), 
				np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]

precision_expected = [1, 2/3]
recall_expected = [1, 0.8]

@pytest.fixture(params=dataset_proba)
def dataset_proba(request):
	y_true = request.param[0]
	y_proba = request.param[1]
	return (y_true, y_proba)


@pytest.fixture(params=dataset_pred)
def dataset_pred(request):
	y_true = request.param[0]
	y_pred = request.param[1]
	return (y_true, y_pred)


# @pytest.fixture(params=roc_expected)
# def roc_expected(request):
# 	fpr = request.param[0]
# 	tpr = request.param[1]
# 	thresholds = request.param[2]
# 	return (fpr, tpr, thresholds)


def test_roc_curve(dataset_proba, roc_expected):
	result = roc_curve(*dataset_proba)
	np.testing.assert_allclose(result, roc_expected)


# @pytest.mark.parametrize('result, expected', [dataset_pred,
# 											    precision_expected])
# def test_precision():
# 	np.testing.assert_allclose(precision(*dataset), )


def test_precision_warn():
	y_true = np.array([0, 1])
	y_pred = np.array([0, 0])
	with pytest.warns(RuntimeWarning):
		precision(y_true, y_pred)


# def test_recall(dataset_pred):
# 	np.testing.assert_allclose(recall(*dataset), 0)

def test_random_splitter():

	# To-do: add more cases
	#
	#	-	number of rows not evenly divisible
	#	-	non-random sampling
	#	-	stratified sampling 

	splitter = Splitter(6, 3, 1)
	X = np.reshape(np.arange(10), (10, 1))
	y = np.arange(10)
	(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = splitter.split(X, y, shuffle=True, random_state=0)
	assert ((X_train == np.array([[2], [8], [4], [9], [1], [6]])).all())
	assert ((y_train == np.array([2, 8, 4, 9, 1, 6])).all())
	assert ((X_valid == np.array([[7], [3], [0]])).all())
	assert ((y_valid == np.array([7, 3, 0])).all())
	assert ((X_test == np.array([[5]])).all())
	assert ((y_test == np.array([5])).all())

def test_roc_curve(dataset_proba):
	result = roc_curve(*dataset_proba)
	np.testing.assert_allclose(result, roc_expected)

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

def test_roc_auc_score():
	y_true = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
	y_proba = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	result = roc_auc_score(y_true, y_proba)
	expected = 0.68
	np.testing.assert_allclose(result, expected)
