from toyml.base import Classifier

class KNearestNeighbours(Classifier):

	def _distance(p1, p2):
		return sum(p1 ** 2 - p2 ** 2)
	