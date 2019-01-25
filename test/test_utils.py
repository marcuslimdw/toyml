from toyml import utils

import pytest

import numpy as np


@pytest.fixture
def probabilities():
    return np.linspace(0, 1, 20, endpoint=False)


def test_normalise(probabilities):
    expected = [0.0,            0.005263157894, 0.010526315789, 0.015789473684,
                0.021052631578, 0.026315789473, 0.031578947368, 0.036842105263,
                0.042105263157, 0.047368421052, 0.052631578947, 0.057894736842,
                0.063157894736, 0.068421052631, 0.073684210526, 0.078947368421,
                0.084210526315, 0.089473684210, 0.094736842105, 0.1]

    np.testing.assert_allclose(utils.normalise(probabilities), expected)


def test_identity(probabilities):
    np.testing.assert_allclose(utils.identity(probabilities), probabilities)


def test_softmax(probabilities):
    expected = [0.029838583826, 0.031368440733, 0.032976735081, 0.034667488444,
                0.036444928585, 0.038313500031, 0.040277875183, 0.042342966004,
                0.044513936295, 0.046796214612, 0.049195507842, 0.051717815466,
                0.054369444567, 0.057157025599, 0.060087528967, 0.063168282456,
                0.066406989554, 0.069811748715, 0.073391073612, 0.077153914420]

    np.testing.assert_allclose(utils.softmax(probabilities), expected)


@pytest.mark.parametrize('metric, expected', [('manhattan', 12.0),
                                              ('euclidean', 56 ** 0.5),
                                              ('chebyshev', 6.0)])
def test_dist(metric, expected):
    p1 = np.array([1, 2, 3])
    p2 = np.array([3, 6, 9])
    np.testing.assert_allclose(utils.dist(p1, p2, metric), expected)
