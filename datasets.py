import numpy as np

from functools import reduce, total_ordering

from itertools import groupby

from operator import add


def _tile(self, value, size):
    return np.tile(A=value, reps=size)


def _choice(self, values, replace, p, size):
    return np.random.choice(a=values, size=size, replace=replace, p=p)


@total_ordering
class Distribution:

    __slots__ = ()

    def __eq__(self, other):
        return (type(self) == type(other) and
                all((getattr(self, attr) == getattr(other, attr) and
                     type(getattr(self, attr)) == type(getattr(other, attr)))
                for attr in self.__slots__))

    def __lt__(self, other):
        return str(self) < str(other)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join('{}={}'.format(slot, getattr(self, slot))
                                                                  for slot in self.__slots__))

    @classmethod
    def make(cls, name, func, slots, defaults, doc=''):
        def init(self, *args, **kwargs):
            if len(args) > len(slots):
                raise TypeError('{} takes up to {} positional arguments but {} were given.'.format(name,
                                                                                                   len(slots),
                                                                                                   len(args)))
            for slot, arg in zip(slots, args):
                setattr(self, slot, arg)

            for kwarg in kwargs:
                if kwarg in slots:
                    if not hasattr(self, kwarg):
                        setattr(self, kwarg, kwargs[kwarg])

                    else:
                        raise TypeError('Multiple values for argument "{}"'.format(slot))

                else:
                    raise TypeError('Unrecognised keyword argument "{}"'.format(kwarg))

            for slot in slots:
                if not hasattr(self, slot):
                    if slot in defaults:
                        setattr(self, slot, defaults[slot])

                    else:
                        raise TypeError('{} missing the positional argument "{}".'.format(name, slot))

        return type(name, (cls,), {'__init__': init, '__slots__': slots, '__doc__': doc, '_FUNC': func})

    def generate(self, shape):
        return self._FUNC(*(getattr(self, slot) for slot in self.__slots__), size=shape)


class RangeDistribution(Distribution):

    __slots__ = ('low', 'high', '_type')

    def __init__(self, low=0.0, high=1.0):
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            if type(low) == type(high):
                self.low = low
                self.high = high
                self._type = type(low)

            else:
                raise TypeError('low (type {}) and high (type {}) must be of the same type.'.format(type(low),
                                                                                                    type(high)))

        else:
            raise TypeError('low (type {}) and high (type {}) must be of type int or float'.format(type(low),
                                                                                                   type(high)))

    def generate(self, shape=(1, 1)):
        if self._type is int:
            return np.random.randint(self.low, self.high + 1, size=shape)

        else:
            return np.random.rand(*shape) * (self.high - self.low) + self.low


class Schema:

    def __init__(self, *feature_groups):

        if feature_groups:
            self.groups = tuple(reduce(add, group) for index, group in groupby(sorted(feature_groups)))

        else:
            self.groups = ()

    def __repr__(self):
        return '[{}]'.format(', '.join(map(str, self.groups)))

    def __getitem__(self, i):
        return self.groups[i]

    def with_features(self, dist, n_features=1):
        return Schema((FeatureGroup(dist, n_features), *self.groups))

    def generate(self, n_rows):
        return np.hstack([group.generate(n_rows) for group in self.groups])


distribution_params = {'ConstantDistribution': {'func': np.tile,
                                                'slots': 'value'},
                       'NormalDistribution':   {'func': np.random.normal,
                                                'slots': ('loc', 'scale')}}

ConstantDistribution = Distribution.make('ConstantDistribution',
                                         func=_tile,
                                         slots=('value',),
                                         defaults={'value': 0})

NormalDistribution = Distribution.make('NormalDistribution',
                                       func=np.random.normal,
                                       slots=('loc', 'scale'),
                                       defaults={'loc': 0,
                                                 'scale': 1})

ChoiceDistribution = Distribution.make('ChoiceDistribution',
                                       func=_choice,
                                       slots=('values', 'replace', 'p'),
                                       defaults={'replace': True,
                                                 'p': None})


class FeatureGroup:

    __slots__ = ('dist', 'size')

    def __init__(self, dist, size):
        self.dist = dist
        self.size = size

    def __add__(self, other):
        if self.dist != other.dist:
            raise ValueError('Cannot combine feature groups with different distribution parameters.')

        else:
            return FeatureGroup(self.dist, self.size + other.size)

    def __ne__(self, other):
        return self.dist != other.dist

    def __eq__(self, other):
        return self.dist == other.dist

    def __lt__(self, other):
        return self.dist < other.dist

    def __le__(self, other):
        return self.dist <= other.dist

    def __gt__(self, other):
        return self.dist > other.dist

    def __ge__(self, other):
        return self.dist >= other.dist

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join('{}={}'.format(slot, getattr(self, slot))
                                                                  for slot in self.__slots__))

    def generate(self, n_rows):
        return self.dist.generate((n_rows, self.size))


def simple(n_points, n_features, seed, proportion=0.5):
    np.random.seed(seed)
    X = Schema(FeatureGroup(RangeDistribution(0.0, 1.0), n_features)).generate(n_points)
    X_error = Schema(FeatureGroup(RangeDistribution(0.0, 1.0), 1)).generate(n_points).sum(axis=1)
    y = (X.sum(axis=1) + X_error < proportion * (n_features - 1)).astype(np.int8)
    return (X, y)
