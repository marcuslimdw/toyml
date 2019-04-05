import pytest

from toyml.linear import LinearRegression

import numpy as np


@pytest.mark.parametrize('n_rows', [100])
@pytest.mark.parametrize('weights', [(2.5, 6.0)])
@pytest.mark.parametrize('noise_factor', [0.0])
def test_simple_noiseless_linear(n_rows, weights, noise_factor):
    model = LinearRegression(learning_rate=0.005, epochs=1000)
    X = np.reshape(np.arange(n_rows), (n_rows, 1))
    y = (X * weights[:-1] + weights[-1]).sum(axis=1) + np.random.rand(n_rows) * noise_factor
    model.fit(X, y)
    np.testing.assert_allclose(model.weights, weights, rtol=5e-1)
