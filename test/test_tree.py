import numpy as np

from toyml.tree import DecisionTreeClassifier
from toyml.evaluation import Splitter

import pytest

@pytest.fixture
def tree():
	return DecisionTreeClassifier()

@pytest.mark.parametrize('output', [np.array([0, 0, 0, 1]),
									np.array([0, 1, 1, 1]),
									np.array([0, 1, 1, 0])])
def test_logic_gates(tree, output):
	data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
	tree.fit(data, output)
	assert (tree.predict(data) == output).all()

@pytest.mark.xfail
def test_simple(tree):
	(X_train, y_train), (X_test, y_test) = simple
	tree.fit(X_train, y_train)
	assert tree.score(X_test, y_test) > 0.8