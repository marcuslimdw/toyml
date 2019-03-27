import pytest

from toyml.tree import DecisionTreeClassifier

import numpy as np


@pytest.mark.parametrize('binning', ['mean', 'ktiles'])
@pytest.mark.parametrize('output', [np.array([0, 0, 0, 1]),
                                    np.array([0, 1, 1, 1]),
                                    np.array([0, 1, 1, 0])])
def test_logic_gates(output, binning):
    model = DecisionTreeClassifier(binning=binning)
    data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    model.fit(data, output)
    assert (model.predict(data) == output).all()


@pytest.mark.parametrize('max_depth', [1, 5, 10])
def test_max_depth(max_depth):
    X = np.reshape(np.arange(max_depth), (max_depth, 1))
    y = np.array([i % 2 for i in range(max_depth)])
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X, y)
    assert max_depth >= model.get_depth()
