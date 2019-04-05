import numpy as np

from .utils import to_checked_X_y, normalise


class Splitter:
    '''Base class for splitters.'''

    def __init__(self, *splits):
        self.splits = np.cumsum(normalise(splits))

    def split(self, X, y, **kwargs):
        X_array, y_array = to_checked_X_y(X, y)
        return self._split(X_array, y_array, **kwargs)


class SimpleSplitter(Splitter):

    def _split(self, X, y):
        yield from RandomSplitter(X, y).split(shuffle=False)


class RandomSplitter(Splitter):

    def _split(self, X, y, shuffle=True, random_state=None):
        length = X.shape[0]
        np.random.seed(random_state)

        if shuffle:
            indices = np.random.permutation(length)
            yield from self._split(X[indices], y[indices], shuffle=False)

        else:
            indices = np.arange(length)
            limits = (np.concatenate(([0], self.splits)) * length).astype(int)
            for start, end in zip(limits, limits[1:]):
                yield (X[indices[start:end]], y[indices[start:end]])


def train_test_split(X, y, train_split=0.7, test_split=0.3, random_state=None):
    '''Split a dataset into training and testing sub-datasets. Quick utility meant to mimic the function of
    `sklearn.model_selection.train_test_split` by wrapping `Splitter`.

    '''
    splitter = RandomSplitter(train_split, test_split)
    (X_train, y_train), (X_test, y_test) = splitter.split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test
