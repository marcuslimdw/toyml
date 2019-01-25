from toyml.base import Classifier


class KNNClassifier(Classifier):

    def __init__(self,
                 k=5,
                 metric='euclidean'):
        self.k = k
        self.metric = metric

    def _fit(self, X, y):
        self._X = X
        self._y = y
        return self

    def _predict_proba(self, X):
        pass
