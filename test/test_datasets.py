from toyml import datasets

import pytest

import numpy as np


def test_range_distribution():
    dist = datasets.RangeDistribution(0, 10)
    result = dist.generate((1000, 1000))
    np.testing.assert_allclose(result.mean(), 4.5, rtol=0.1)
    np.testing.assert_allclose(result.mean(axis=0), 4.5, rtol=0.1)
    np.testing.assert_allclose(result.mean(axis=1), 4.5, rtol=0.1)


class TestEquality:

    @pytest.mark.parametrize('value_1', [0, 1])
    @pytest.mark.parametrize('value_2', [0, 1])
    def test_constant(self, value_1, value_2):
        first = datasets.ConstantDistribution(value_1)
        second = datasets.ConstantDistribution(value_2)
        expected = (value_1 == value_2)
        assert (first == second) == expected

    @pytest.mark.parametrize('value_1_low, value_1_high', [(0, 1), (5, 10), (3, 6)])
    @pytest.mark.parametrize('value_2_low, value_2_high', [(0, 1), (5, 10), (6, 9)])
    def test_range(self, value_1_low, value_1_high, value_2_low, value_2_high):
        first = (datasets.RangeDistribution(value_1_low, value_1_high))
        second = datasets.RangeDistribution(value_2_low, value_2_high)
        expected = (value_1_low == value_2_low) and (value_1_high == value_2_high)
        assert (first == second) == expected
