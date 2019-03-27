from toyml import metrics

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
        result = metrics.roc_curve(*data)
        np.testing.assert_allclose(result, expected)

    else:
        with pytest.raises(type(expected)):
            metrics.roc_curve(*data)


@pytest.mark.parametrize('data, expected', zip(dataset_pred, precision_expected))
def test_precision(data, expected):
    if not isinstance(expected, Warning):
        result = metrics.precision(*data)
        np.testing.assert_allclose(result, expected)

    else:
        with pytest.warns(type(expected)):
            assert np.isnan(metrics.precision(*data))


@pytest.mark.parametrize('data, expected', zip(dataset_pred, recall_expected))
def test_recall(data, expected):
    if not isinstance(expected, Warning):
        result = metrics.recall(*data)
        np.testing.assert_allclose(result, expected)
    else:
        with pytest.warns(type(expected)):
            assert np.isnan(metrics.recall(*data))


@pytest.mark.parametrize('data, expected', zip(dataset_proba, roc_auc_score_expected))
def test_roc_auc_score(data, expected):
    if not isinstance(expected, Warning):
        result = metrics.roc_auc_score(*data)
        np.testing.assert_allclose(result, expected)

    else:
        with pytest.warns(type(expected)):
            assert np.isnan(metrics.roc_auc_score(*data))


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
    np.testing.assert_allclose(metrics.auc(x, y), expected, atol=1e-06)
