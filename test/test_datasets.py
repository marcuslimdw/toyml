from toyml.datasets import ConstantDistribution, NormalDistribution, RangeDistribution, ChoiceDistribution, \
                           FeatureGroup, Schema

import pytest

import numpy as np


@pytest.fixture
def actual_dist_size():
    return (1000, 1000)


class TestActualDistribution:

    @pytest.mark.parametrize('value', [-10.0, 0, 10, 100.0])
    def test_constant(self, value, actual_dist_size):
        dist = ConstantDistribution(value=value)
        result = dist.generate(actual_dist_size)
        np.testing.assert_allclose(result, value)

    @pytest.mark.parametrize('low, high', [(0, 1),
                                           (0.0, 1.0),
                                           (0, 10),
                                           (0.0, 10.0)])
    def test_range(self, low, high, actual_dist_size):
        dist = RangeDistribution(low=low, high=high)
        result = dist.generate(actual_dist_size)
        tolerance = {'atol': 0.05, 'rtol': 0.075}
        for axis in (0, 1, None):
            np.testing.assert_allclose(result.mean(axis=axis), (low + high) / 2, **tolerance)

    @pytest.mark.parametrize('loc, scale', [(0.0, 1.0),
                                            (10.0, 2.0),
                                            (-10.0, 3.0)])
    def test_normal(self, loc, scale, actual_dist_size):
        dist = NormalDistribution(loc=loc, scale=scale)
        result = dist.generate(actual_dist_size)
        tolerance = {'atol': 0.135, 'rtol': 0.135}
        for axis in (0, 1, None):
            np.testing.assert_allclose(result.mean(axis=axis), loc, **tolerance)
            np.testing.assert_allclose(result.std(axis=axis), scale, **tolerance)

    @pytest.mark.xfail
    @pytest.mark.parametrize('values, p', [(['a', 'b', 'c'], None),
                                           (['a', 'b', 'c'], [0.5, 0.25, 0.25]),
                                           (['a', 'b', 'c'], [0.75, 0.25, 0.0])])
    def test_choice(self, values, p, actual_dist_size):
        dist = ChoiceDistribution(values=values, replace=True, p=p)
        result = dist.generate(actual_dist_size)
        tolerance = {'atol': 0.125, 'rtol': 0.1}
        np.assert_allclose(result, p, **tolerance)


class TestEquality:

    @pytest.mark.parametrize('value_1', [0, 1])
    @pytest.mark.parametrize('value_2', [0, 1])
    def test_constant(self, value_1, value_2):
        first = ConstantDistribution(value_1)
        second = ConstantDistribution(value_2)
        expected = (value_1 == value_2)
        assert (first == second) == expected

    @pytest.mark.parametrize('value_1_low, value_1_high', [(0, 1), (5, 10), (3, 6)])
    @pytest.mark.parametrize('value_2_low, value_2_high', [(0, 1), (5, 10), (6, 9)])
    def test_range(self, value_1_low, value_1_high, value_2_low, value_2_high):
        first = (RangeDistribution(value_1_low, value_1_high))
        second = RangeDistribution(value_2_low, value_2_high)
        expected = (value_1_low == value_2_low) and (value_1_high == value_2_high)
        assert (first == second) == expected


@pytest.mark.parametrize('dist, valid', [(ConstantDistribution, True),
                                         (NormalDistribution, True),
                                         (RangeDistribution, True),
                                         (ChoiceDistribution, False)])
def test_default_arguments(dist, valid):
    if valid:
        dist()

    else:
        with pytest.raises(TypeError):
            dist()


@pytest.mark.parametrize('first_size, second_size', [(1, 1),
                                                     (3, 3),
                                                     (0, 1)])
@pytest.mark.parametrize('first, second, valid', [(ConstantDistribution(), ConstantDistribution(), True),
                                                  (ConstantDistribution(0), ConstantDistribution(0.0), False),
                                                  (NormalDistribution(), NormalDistribution(), True),
                                                  (NormalDistribution(0, 1), NormalDistribution(0, 2), False),
                                                  (RangeDistribution(), RangeDistribution(), True),
                                                  (RangeDistribution(), RangeDistribution(1, 2), False),
                                                  (RangeDistribution(0, 1), RangeDistribution(0.0, 1.0), False)])
def test_feature_group(first_size, second_size, first, second, valid):
    if valid:
        assert (FeatureGroup(first, first_size) + FeatureGroup(second, second_size)).size == first_size + second_size

    else:
        with pytest.raises(ValueError):
            FeatureGroup(first, first_size) + FeatureGroup(second, second_size)


@pytest.mark.parametrize('data, expected', [([FeatureGroup(ConstantDistribution(), 3)] * 3,
                                             [FeatureGroup(ConstantDistribution(), 9)]),

                                            ([FeatureGroup(ConstantDistribution(), 3),
                                              FeatureGroup(NormalDistribution(), 3),
                                              FeatureGroup(ConstantDistribution(), 3)],
                                             [FeatureGroup(ConstantDistribution(), 6),
                                              FeatureGroup(NormalDistribution(), 3)]),

                                            ([FeatureGroup(ConstantDistribution(), 3),
                                              FeatureGroup(NormalDistribution(), 3),
                                              FeatureGroup(ConstantDistribution(), 3),
                                              FeatureGroup(RangeDistribution(), 3),
                                              FeatureGroup(ConstantDistribution(), 3),
                                              FeatureGroup(NormalDistribution(), 3)],

                                             [FeatureGroup(ConstantDistribution(), 9),
                                              FeatureGroup(NormalDistribution(), 6),
                                              FeatureGroup(RangeDistribution(), 3)])])
def test_schema(data, expected):
    n_points = 10
    schema = Schema(*data)
    assert sorted(schema.groups) == sorted(expected)
    assert schema.generate(n_points).shape == (n_points, sum(group.size for group in expected))
