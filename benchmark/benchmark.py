import numpy as np

from time import clock

from contextlib import contextmanager

from toyml.linear import LinearRegression as TLR
from sklearn.linear_model import LinearRegression as SLR

from toyml.tree import DecisionTreeClassifier as TDTC
from sklearn.tree import DecisionTreeClassifier as SDTC

from toyml.evaluation import train_test_split


def simple(n_points, n_features, seed, noise_scale=0.2, threshold=0.5):
    np.random.seed(seed)
    X = np.random.rand(n_points, n_features)
    X_error = np.random.rand(n_points, n_features) * noise_scale
    y = ((X + X_error).sum(axis=1) >= threshold * (n_features)).astype(np.int8)
    return (X, y)


@contextmanager
def timer(current):
    time_now = clock()
    yield
    print('{} took {:.2f} seconds.'.format(current, clock() - time_now))


tlinr = TLR(epochs=1000)
slinr = SLR()

tdtc = TDTC(max_depth=8)
sdtc = SDTC()

X = np.random.rand(10000, 10)
y = (X * np.random.rand(10)).sum(axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

with timer('toyml LinearRegression.fit'):
    tlinr.fit(X_train, y_train)

with timer('sklearn LinearRegression.fit'):
    slinr.fit(X_train, y_train)

with timer('toyml LinearRegression.predict'):
    print('toyml scored {}.'.format(tlinr.score(X_test, y_test)))

with timer('sklearn LinearRegression.predict'):
    print('sklearn scored {}.'.format(slinr.score(X_test, y_test)))

X_train, y_train = simple(20000, 20, 0)
X_test, y_test = simple(10000, 20, 0)

with timer('toyml DecisionTreeClassifier.fit'):
    tdtc.fit(X_test, y_test)

with timer('sklearn DecisionTreeClassifier.fit'):
    sdtc.fit(X_train, y_train)

with timer('toyml DecisionTreeClassifier.predict'):
    print('toyml scored {}.'.format(tdtc.score(X_test, y_test)))

with timer('sklearn DecisionTreeClassifier.predict'):
    print('sklearn scored {}.'.format(sdtc.score(X_test, y_test)))
