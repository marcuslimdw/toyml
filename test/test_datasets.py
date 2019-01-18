import pytest

import numpy as np

from toyml.datasets import *

def test_range_distribution():
	dist = RangeDistribution('int', 0, 10)
		result = dist.generate((1000, 1000))
		np.testing.assert_allclose(result.mean(), 4.5, rtol=0.1)
		np.testing.assert_allclose(result.mean(axis=0), 4.5, rtol=0.1)
		np.testing.assert_allclose(result.mean(axis=1), 4.5, rtol=0.1)

def test_range_distribution_wrong_type():
	with pytest.raises(ValueError):
		RangeDistribution('str')