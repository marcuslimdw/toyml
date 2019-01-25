import numpy as np

_METRICS = {'manhattan': lambda p1, p2: sum(abs(p1 - p2)),
            'euclidean': lambda p1, p2: sum((p1 - p2) ** 2) ** 0.5,
            'chebyshev': lambda p1, p2: max(abs(p1 - p2))}


def check_X_y(X, y):
    check_shape(X, 2)
    check_shape(y, 1)
    check_alignment(X, y)


def check_shape(array_like, expected_dim):
    array_dim = len(np.array(array_like).shape)
    if array_dim != expected_dim:

        if array_dim == 1 and expected_dim == 2:
            raise ValueError(('Got a 1D array-like object where a 2D array was expected. If your sample has one'
                              'observation or one feature, reshape it as a 2D array with one row or column'
                              'accordingly.'))

        else:
            raise ValueError('Got a {}D array-like object where a {}D array was expected.'
                             .format(array_dim, expected_dim))


def check_alignment(X, y):
    if X.shape[0] != y.size:
        raise ValueError('The dimensions of X {} and y {} do not match.'.format(X.shape, y.shape))


def normalise(x):
    array_x = np.array(x)
    return array_x / sum(array_x)


def identity(x):
    return x


def softmax(x):
    return normalise(np.exp(x))


def dist(p1,
         p2,
         metric='euclidean'):

    return _METRICS[metric](p1, p2)
