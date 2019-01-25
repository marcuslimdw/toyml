from toyml import tree, datasets

import pytest

import numpy as np


@pytest.fixture
def dataset():
    return datasets.simple(20000, 20, 0)


@pytest.mark.parametrize('binning', ['mean', 'ktiles'])
@pytest.mark.parametrize('output', [np.array([0, 0, 0, 1]),
                                    np.array([0, 1, 1, 1]),
                                    np.array([0, 1, 1, 0])])
def test_logic_gates(output, binning):
    model = tree.DecisionTreeClassifier(binning=binning)
    data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    model.fit(data, output)
    assert (model.predict(data) == output).all()


@pytest.mark.slow
@pytest.mark.parametrize('model', [tree.DecisionTreeClassifier(max_depth=0),
                                   tree.DecisionTreeClassifier(max_depth=3),
                                   tree.DecisionTreeClassifier(max_depth=20),
                                   tree.DecisionTreeClassifier(max_depth=None)])
def test_max_depth(model, dataset):
    model.fit(*dataset)
    if model.max_depth is not None:
        assert model.max_depth >= model.get_depth()
