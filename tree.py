import numpy as np

from toyml.base import Classifier

from functional import flatten

from collections import namedtuple


def _gini(y):
    '''
    Calculate the score of a dataset of labels based on Gini impurity.

    Randomly choose one element from the dataset and assign to it a random label, based on the distribution of labels in
    the dataset. The probability that this random label is wrong is the Gini impurity. To give a metric that increases
    with quality, the score returned is 1 - Gini impurity.

    Parameters
    ----------

    y: 1D array
        The dataset to calculate the Gini impurity of.

    Returns
    ----------

    score: float
        The calculated Gini impurity-based score of `y`.

    Examples
    ----------

    Given `y = [1, 1, 1, 1, 1]`, a randomly chosen element can only be 1. Randomly choosing a label from the dataset's
    distribution of labels can also only give 1. The probability that the two do not match, and therefore the Gini
    impurity, is 0, giving a score of 1.0 - 0.0 = 1.0.

    Given `y = [1, 1, 2, 2, 3]`, the probabilities of choosing each unique label are as follows:

    1: 0.4
    2: 0.4
    3: 0.2

    In the case of a randomly chosen label of 1, there is a 1.0 - 0.6 = 0.4 probability that the correct label is 2 or
    3, for a total probability of 0.6 * 0.4 = 0.24 of making a mistake. This similarly applies for class 2. For class 3,
    the probability of a mistake is 0.2 * 0.8 = 0.16. Summing all of these probabilities up, the Gini impurity of the
    dataset is 0.64, and therefore the score is 1.0 - 0.64 = 0.36.
    '''

    p = np.unique(y, return_counts=True)[1] / y.size
    return 1 - sum(p * (1 - p))


class Node:

    Split = namedtuple('Split', ('feature_index', 'on', 'weighted_quality', 'n_samples'))

    def __init__(self, head, X, y, depth):
        self._head = head
        self._depth = depth

        if len(X) < self._head.min_samples_split or self._too_deep():
            pre_split_terminate = True

        else:
            pre_split_terminate = False
            self._quality = self._head._quality_func(y)
            self._split = self._get_best_split(X, y)

        # Is there a better way to do this?

        if pre_split_terminate or self._split_terminate():
            label_counts = dict(zip(*np.unique(y, return_counts=True)))
            self._probabilities = np.array([label_counts[label] / y.size
                                            if label in label_counts
                                            else 0
                                            for label in self._head.labels])
            self._leaf = True

        else:
            self._left = Node(self._head, *self._split_left(X, y), depth=self._depth + 1)
            self._right = Node(self._head, *self._split_right(X, y), depth=self._depth + 1)
            self._leaf = False

    def predict_proba(self, X):
        '''
        Predict the probabilities that each observation in a dataset has of belonging to each label.

        Parameters
        ----------

        X: 2D array
            The dataset containing observations.

        Returns
        ----------

        probabilities: 2D array
            The probabilities that each observation contained in `X` has of belonging to each label.
        '''
        return np.vstack(self._get_probabilities(x) for x in X)

    def _too_deep(self):
        return self._head.max_depth is not None and self._depth >= self._head.max_depth

    def _split_terminate(self):
        '''Check if splitting should terminate, based on the following criteria, in order:

        1. There are no remaining possible splits; or

        2. The quality of the current node is higher than the weighted quality of the best possible split.

        Returns
        ----------

        terminate: bool
            Whether or not to terminate splitting.

        '''

        # `or` here is necessary to force short-circuit evaluation (otherwise the quality comparison might fail).

        return (self._split is None or
                self._quality > self._split.weighted_quality)

    def _weighted_quality(self, left_y, right_y):
        '''
        Calculate the weighted quality of a split by applying the quality function to each sub-dataset and taking the
        mean weighted by the size the sub-datasets.

        Parameters
        ----------

        left_y: 1D array
            The sub-dataset containing the labels in the left split.

        right_y: 1D array
            The sub-dataset containing the labels in the right split.

        Returns
        ----------

        quality: float
            The average of the quality metric for both sub-datasets, weighted by the sizes of the sub-datasets.
        '''

        left_size = left_y.size
        right_size = right_y.size
        total_size = left_size + right_size
        return (self._head._quality_func(left_y) * left_size +
                self._head._quality_func(right_y) * right_size) / total_size

    def _splits(self, x, y, j):
        '''
        Generate all possible splits of a feature on unique values of associated labels after binning.

        Parameters
        ----------

        x: 1D array
            The sub-dataset containing observations of a single feature.

        y: 1D array
            The labels corresponding to the observations in `x`.

        j: int
            The index of the feature contained in `x`.

        k: int, default 10
            The number of quantiles to divide the sub-dataset into.

        Yields
        ----------

        split: dict
            A candidate split containing the following keys:

            feature_index: int
                The index of the feature to split on, equal to `j`.

            on: ordered type
                The value to split on. Values equal to or lower than this value will fall into the left split, while
                values greater will fall into the right split.

            weighted_quality: double
                The weighted quality of the split as calculated by `_weighted_quality`.

            n_samples: tuple
                The number of samples in the left and right splits.
        '''

        if self._head.binning == 'mean':
            uniques = np.unique(x)
            bin_edges = np.stack((uniques[:-1], uniques[1:])).mean(axis=0)

        elif self._head.binning == 'ktiles':
            bin_edges = np.unique(np.quantile(x, np.linspace(0, 1, self._head.k + 1, endpoint=False)))[1:]

        else:
            raise RuntimeError('Got unknown value "{}" for "binning": expected one of {}.'.format(self._head.binning,
                                                                                                  self._head._BINNING))

        for bin_edge in bin_edges:
            left_y = y[x < bin_edge]
            right_y = y[x >= bin_edge]

            yield self.Split(feature_index=j,
                             on=bin_edge,
                             weighted_quality=self._weighted_quality(left_y, right_y),
                             n_samples=(left_y.size, right_y.size))

    def _get_best_split(self, X, y):
        '''
        Get the best feature and value to split on, based on the quality metric.

        Parameters
        ----------

        X: 2D array
            The dataset containing observations of a number of features.

        y: 1D array
            The labels corresponding to the observations in `X`.

        Returns
        ----------

        best_split: dict
            The dictionary containing the parameters for the best possible split. See `_splits`.
        '''

        def is_binary(split):
            return min(split.n_samples) > 0

        candidates = (filter(is_binary, self._splits(X[:, j], y, j)) for j in range(X.shape[1]))
        try:
            best_split = max(flatten(candidates), key=lambda split: split.weighted_quality)

        except ValueError:
            best_split = None

        return best_split

    def _split_left(self, X, y):
        '''
        Get the left split of a dataset, based on the current node's splitting criteria.

        Parameters
        ----------

        X: 2D array
            The dataset containing observations of a number of features.

        y: 1D array
            The labels corresponding to the observations in `X`.

        Returns
        ----------

        X_split: 2D array
            The sub-dataset containing observations that fall on the left split of the current node.

        y_split: 1D array
            The labels corresponding to the observations in `X_split`.
        '''

        index = X[:, self._split.feature_index] <= self._split.on
        return (X[index], y[index])

    def _split_right(self, X, y):
        '''
        Get the right split of a dataset, based on the current node's splitting criteria.

        Parameters
        ----------

        X: 2D array
            The dataset containing observations of a number of features.

        y: 1D array
            The labels corresponding to the observations in `X`.

        Returns
        ----------

        X_split: 2D array
            The sub-dataset containing observations that fall on the right split of the current node.

        y_split: 1D array
            The labels corresponding to the observations in `X_split`.
        '''

        index = X[:, self._split.feature_index] > self._split.on
        return (X[index], y[index])

    def _get_probabilities(self, x):
        '''
        Get the probabilities that an observation has of belonging to each label. If the current node is not a leaf,
        traverse its child nodes recursively until a leaf whose criteria x satisfies is found.

        Parameters
        ----------

        x: 1D array
            The sub-dataset containing a single observation.

        Returns
        ----------

        probabilities: 1D array
            The probabilities that the observation contained in `x` has of belonging to each label.
        '''

        if self._leaf:
            return self._probabilities

        else:
            if x[self._split.feature_index] <= self._split.on:
                return self._left._get_probabilities(x)

            else:
                return self._right._get_probabilities(x)


class DecisionTreeClassifier(Classifier):
    '''
    A decision tree classifier.

    Parameters
    ----------

    quality_func: callable
        A callable object to determine the quality of a split. It should take a 1D array containing labels and return a
        number representing the split's quality, with higher numbers being better.

    max_depth: int
        The maximum depth to which to grow the tree.
    '''

    _BINNING = ('mean', 'ktiles')

    def __init__(self,
                 quality_func=_gini,
                 binning='ktiles',
                 k=5,
                 max_depth=5,
                 min_samples_split=1):
        self._quality_func = quality_func
        self.binning = binning
        self.k = k
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _fit(self, X, y):
        self.labels = np.unique(y)
        self.base = Node(self, X, y, 0)
        return self

    def _predict_proba(self, X):
        return self.base.predict_proba(X)

    def get_depth(self):
        def traverse(node, current):
            if node._leaf:
                return current

            else:
                return max(traverse(node._left, current + 1), traverse(node._right, current + 1))

        return traverse(self.base, 0)
