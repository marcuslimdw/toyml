from toyml import evaluation as ev

import numpy as np

import pytest

dataset_proba = [[np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                  np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])],

                 [np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
                  np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])],

                 [np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                  np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])],

                 [np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                  np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])],

                 [np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1]),
                  np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]]

dataset_pred = [(y_true, (y_proba >= 0.5).astype(np.int8)) for (y_true, y_proba) in dataset_proba]

roc_expected = [[np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0]),
                 np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])],

                [np.array([1.0, 1.0, 0.0]),
                 np.array([1.0, 1.0, 0.0]),
                 np.array([0.0, 0.8, 1.0])],

                ValueError(),

                ValueError(),

                [np.array([1.0, 0.8, 0.8, 0.6, 0.4, 0.4, 0.2, 0.2, 0.2, 0.0, 0.0]),
                 np.array([1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.6, 0.4, 0.2, 0.2, 0.0]),
                 np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]]

precision_expected = [1.0,
                      0.5,
                      0.0,
                      1.0,
                      0.6
                      ]

recall_expected = [1.0,
                   1.0,
                   RuntimeWarning(),
                   0.5,
                   0.6
                   ]

roc_auc_score_expected = [1.0,
                          0.5,
                          RuntimeWarning(),
                          RuntimeWarning(),
                          0.68
                          ]


@pytest.mark.parametrize('data, expected', zip(dataset_proba, roc_expected))
def test_roc_curve(data, expected):
    if not isinstance(expected, Exception):
        result = ev.roc_curve(*data)
        np.testing.assert_allclose(result, expected)

    else:
        with pytest.raises(type(expected)):
            ev.roc_curve(*data)


@pytest.mark.parametrize('data, expected', zip(dataset_pred, precision_expected))
def test_precision(data, expected):
    if not isinstance(expected, Warning):
        result = ev.precision(*data)
        np.testing.assert_allclose(result, expected)

    else:
        with pytest.warns(type(expected)):
            assert np.isnan(ev.precision(*data))


@pytest.mark.parametrize('data, expected', zip(dataset_pred, recall_expected))
def test_recall(data, expected):
    if not isinstance(expected, Warning):
        result = ev.recall(*data)
        np.testing.assert_allclose(result, expected)
    else:
        with pytest.warns(type(expected)):
            assert np.isnan(ev.recall(*data))


@pytest.mark.parametrize('data, expected', zip(dataset_proba, roc_auc_score_expected))
def test_roc_auc_score(data, expected):
    if not isinstance(expected, Warning):
        result = ev.roc_auc_score(*data)
        np.testing.assert_allclose(result, expected)

    else:
        with pytest.warns(type(expected)):
            assert np.isnan(ev.roc_auc_score(*data))


# def test_random_splitter():

# 	# To-do: add more cases
# 	#
# 	#	-	number of rows not evenly divisible
# 	#	-	non-random sampling
# 	#	-	stratified sampling

# 	splitter = Splitter(6, 3, 1)
# 	y = np.arange(10)
# 	X = np.reshape(y, (10, 1))
# 	(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = splitter.split(X, y, shuffle=True, random_state=0)
# 	np.testing.assert_array_equal(np.reshape(X_train, (X_train.size)), y_train)
# 	np.testing.assert_array_equal(np.reshape(X_valid, (X_valid.size)), y_valid)
# 	np.testing.assert_array_equal(np.reshape(X_test, (X_test.size)), y_test)

@pytest.mark.parametrize('x, y, expected', [((0, 1), (0, 0), 0),
                                            ((0, 0, 1, 1), (0, 1, 1, 0), 1),
                                            ((0, 1, 1), (0, 1, 0), 0.5),
                                            ((0, 0, 1, 1), (0, -1, -1, 0), -1),
                                            ((0.0, 0, 0.5, 0.5, 1.0, 1.0), (0.0, 1.0, 1.0, -1.0, -1.0, 0), 0),
                                            ((0.0, 0.0, 0.5, 1.0, 1.0), (0.0, 1.0, 1.0, -1.0, 0.0), 0.5),
                                            (np.linspace(0, 2 * np.pi, 400), np.sin(np.linspace(0, 2 * np.pi, 400)), 0),
                                            (np.linspace(0, 2 * np.pi, 400), np.cos(np.linspace(0, 2 * np.pi, 400)), 0),
                                            (np.linspace(0, 1, 400), np.exp(np.linspace(0, 1, 400)), np.e - 1)])
def test_auc(x, y, expected):
    np.testing.assert_allclose(ev.auc(x, y), expected, atol=1e-06)
