import numpy as np

from toyml import evaluation as ev


def test_random_splitter():

    # To-do: add more cases
    #
    #   -   number of rows not evenly divisible
    #   -   non-random sampling
    #   -   stratified sampling

    splitter = ev.RandomSplitter(6, 3, 1)
    y = np.arange(10)
    X = np.reshape(y, (10, 1))
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = splitter.split(X, y, shuffle=True, random_state=0)
    np.testing.assert_array_equal(np.reshape(X_train, (X_train.size)), y_train)
    np.testing.assert_array_equal(np.reshape(X_valid, (X_valid.size)), y_valid)
    np.testing.assert_array_equal(np.reshape(X_test, (X_test.size)), y_test)
