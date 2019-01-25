from time import clock

from contextlib import contextmanager

from toyml.tree import DecisionTreeClassifier as TDTC
from sklearn.tree import DecisionTreeClassifier as SDTC

from datasets import simple

time_now = 0.0


@contextmanager
def timer(current):
    time_now = clock()
    yield
    print('{} took {:.2f} seconds.'.format(current, clock() - time_now))


X_train, y_train = simple(2000, 6, 0)
X_test, y_test = simple(1000, 6, 0)

toymldtc = TDTC(max_depth=5)
skldtc = SDTC()

with timer('toyml DecisionTreeClassifier.fit'):
    toymldtc.fit(X_test, y_test)

with timer('sklearn DecisionTreeClassifier.fit'):
    skldtc.fit(X_train, y_train)

with timer('toyml DecisionTreeClassifier.predict'):
    print('toyml scored {}.'.format(toymldtc.score(X_test, y_test)))

with timer('sklearn DecisionTreeClassifier.predict'):
    print('sklearn scored {}.'.format(skldtc.score(X_test, y_test)))
