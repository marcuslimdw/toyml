import unittest

import numpy as np

from toyml.utils import dist

class TestUtils(unittest.TestCase):

	def test_dist(self):
		p1 = np.array([1, 2, 3])
		p2 = np.array([3, 6, 9])
		self.assertAlmostEqual(dist(p1, p2, 'manhattan'), 
							   12.0)
		self.assertAlmostEqual(dist(p1, p2, 'euclidean'),
							   56 ** 0.5)
		self.assertAlmostEqual(dist(p1, p2, 'chebyshev'), 
							   6.0)

unittest.main()
