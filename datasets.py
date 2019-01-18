import numpy as np

from collections import namedtuple

from operator import itemgetter

from toyml.utils import check_shape, normalise
from toyml.evaluation import Splitter

# TO-DO: think about how to handle allowed dtypes - should there be automatic upcasting?

Feature = namedtuple('Feature', ('dtype', 'dist'))

class Distribution:

	_ALLOWED = ('int', 'float', 'str')

	def __init__(self, dtype):
		if dtype not in self._ALLOWED:
			raise ValueError('{} expected a dtype in {} but {} was passed.'.format(type(self).__name__,
																				   self._ALLOWED,
																				   dtype))
		self._dtype = dtype

	def generate(self, shape):
		raise NotImplementedError


class ConstantDistribution(Distribution):

	def __init__(self, dtype, value):
		super().__init__(dtype)
		self._value = value

	def generate(self, shape=(1, 1)):
		return np.tile(self._value, shape)


class ChoiceDistribution(Distribution):

	def __init__(self, dtype, values, replace=True, p=None):
		super().__init__(dtype)
		check_shape(values, 1)
		self._values = values
		self._replace = replace
		self._p = p

	def generate(self, shape=(1, 1)):
		return np.random.choice(self._values, size=shape, replace=self._replace, p=self._p)


class NormalDistribution(Distribution):

	_ALLOWED = ('float')

	def __init__(self, dtype, loc=0.0, scale=1.0):
		super().__init__(dtype)
		self._loc = loc
		self._scale = scale

	def generate(self, shape=(1, 1)):
		return np.random.normal(loc=self._loc, scale=self._scale)


class RangeDistribution(Distribution):

	_ALLOWED = ('int', 'float')

	def __init__(self, dtype, low=None, high=None):
		super().__init__(dtype)
		if (low is None) ^ (high is None):
			raise ValueError('Either both or neither of low and high must be None.')

		elif low is None:
			if dtype == 'int':
				self._low = 0
				self._high = 2

			else:
				self._low = 0
				self._high = 1

		else:
			self._low = low
			self._high = high

	def generate(self, shape=(1, 1)):
		if self._dtype == 'int':
			return np.random.randint(self._low, self._high, size=shape)

		else:
			return np.random.rand(*shape) * (self._high - self._low) + self._low


class Schema:

	def __init__(self, features=None):
		if features is None:
			self._features = ()

		else:
			self._features = features

	def with_features(self, dtype, dist, n_features=1):
		return Schema(self._features + tuple(Feature(dtype, dist) for i in range(n_features)))

	def groups(self):
		feature_types = sorted(set(self._features))
		for feature_type in feature_types:
			yield FeatureGroup(feature_type, self._features.count(feature_type))


class FeatureGroup:

	def __init__(self, feature_type, size):
		self.dtype = feature_type.dtype
		self.dist = feature_type.dist
		self.size = size

	def generate(self, n_rows=1):
		return self.dist.generate((n_rows, self.size))


class DataGenerator:

	def __init__(self, schema):
		self.schema = schema

	def generate(self, n_rows):
		return np.hstack([group.generate(n_rows) for group in self.schema.groups()])
