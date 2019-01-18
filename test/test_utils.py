import pytest

import numpy as np

from toyml.utils import dist

@pytest.mark.parametrize('metric, expected', [('manhattan', 12.0),
											  ('euclidean', 56 ** 0.5),
											  ('chebyshev', 6.0)])
def test_dist(metric, expected):
	p1 = np.array([1, 2, 3])
	p2 = np.array([3, 6, 9])
	np.testing.assert_allclose(dist(p1, p2, metric), expected)