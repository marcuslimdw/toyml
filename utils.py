import numpy as np

_METRICS = {'manhattan': lambda p1, p2: sum(abs(p1 - p2)),
            'euclidean': lambda p1, p2: sum((p1 - p2) ** 2) ** 0.5,
            'chebyshev': lambda p1, p2: max(abs(p1 - p2))}


def to_checked_X_y(X, y):
    X_array = to_checked_array(X, 2)
    y_array = to_checked_array(y, 1)
    check_alignment(X_array, y_array)
    return (X_array, y_array)


def to_checked_array(array_like, expected_dim):
    array = np.array(array_like)
    array_dim = len(array.shape)
    if array_dim != expected_dim:

        if array_dim == 1 and expected_dim == 2:
            raise ValueError(('Got a 1D array-like object where a 2D array was expected. If your sample has one'
                              'observation or one feature, reshape it as a 2D array with one row or column'
                              'accordingly.'))

        else:
            raise ValueError('Got a {}D array-like object where a {}D array was expected.'
                             .format(array_dim, expected_dim))

    return array


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


def flatten(iterable, levels=1):
    '''
    Make an iterator over the result of applying `levels` levels of flattening to `iterable`.

    Parameters
    ----------
    iterable: iterable object
        The iterable to flatten.

    levels: int
        The class or classes of Exception to ignore.

    Returns
    ----------
    flattened: iterator

    Notes
    ----------
    Assumes that nesting is even; in other words, that each the sub-iterables in each level are further nested to the
    same depth.

    While `levels` accepts floating point types, this is unsupported (may cause unexpected behaviour in some rare cases
    due to floating point rounding).
    '''
    if levels <= 0:
        yield from iterable

    else:
        for sub_iterable in iterable:
            yield from flatten(sub_iterable, levels=levels - 1)
