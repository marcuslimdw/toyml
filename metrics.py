import numpy as np

from warnings import warn


def accuracy(y_true, y_pred):
    '''
    Calculate the accuracy of a classification prediction, equal to the fraction of samples correctly classified.

    Parameters
    ----------

    y_true: 1D array
        The actual labels of a number of observations.

    y_pred: 1D array
        The predicted labels of a number of observations.

    Returns
    ----------

    accuracy: float
        The accuracy of the prediction in `y_pred`.
    '''
    return (y_true == y_pred).sum() / y_true.size


def precision(y_true, y_pred, which=1):
    '''
    Calculate the precision of a classification prediction for a particular class. The precision for a class is equal
    to the proportion of observations predicted to be of a particular class that are actually of that class, i.e. true
    positives / predicted positives.

    Parameters
    ----------

    y_true: 1D array
        The actual labels of a number of observations.

    y_pred: 1D array
        The predicted labels of a number of observations.

    which: int
        The class for which to calculate precision.

    Returns
    ----------

    precision: float
        The precision of the prediction in `y_pred` for the class `which`.
    '''
    true_positives = ((y_true == which) & (y_pred == which)).sum()
    predicted_positives = (y_pred == which).sum()
    if predicted_positives != 0:
        return true_positives / predicted_positives

    else:
        warn(RuntimeWarning('Precision is undefined when there are no predicted positives. Returning np.nan.'))
        return np.nan


def recall(y_true, y_pred, which=1):
    '''
    Calculate the recall of a classification prediction for a particular class. The recall for a class is equal to the
    proportion of observations actually of a class that were also predicted to be of that class, i.e. true positives
    divided by all positives.

    Parameters
    ----------

    y_true: 1D array
        The actual labels of a number of observations.

    y_pred: 1D array
        The predicted labels of a number of observations.

    which: int
        The class for which to calculate recall.

    Returns
    ----------

    recall: float
        The recall of the prediction in `y_probared` for the class `which`.

    '''
    true_positives = ((y_true == which) & (y_pred == which)).sum()
    all_positives = (y_true == which).sum()
    if all_positives != 0:
        return true_positives / all_positives
    else:
        warn(RuntimeWarning('Recall is undefined when there are no true positives. Returning np.nan.'))
        return np.nan


def mean_squared_error(y_true, y_pred):
    '''
    Calculate the mean squared error of a regression prediction, equal to the arithmetic mean of the square of the
    difference between each ground truth value and the corresponding prediction.

    Parameters
    ----------

    y_true: 1D array
        The actual values of a number of observations.

    y_pred: 1D array
        The predicted values of a number of observations.

    Returns
    ----------

    mse: float
        The mean squared error of the prediction in `y_pred`.
    '''

    return ((y_true - y_pred) ** 2).sum() / y_true.size


def r_squared(y_true, y_pred):
    unexplained_variance = mean_squared_error(y_true, y_pred)
    total_variance = mean_squared_error(y_true, y_true.mean())
    return 1 - (unexplained_variance / total_variance)


def roc_curve(y_true, y_proba):
    '''
    Generate the points needed to plot the Receiver Operating Characteristic curve.

    Parameters
    ----------

    y_true: 1D array
        The actual labels of a number of observations.

    y_proba: 1D array
        The predicted probabilities of each observation belonging to each class.

    Returns
    ----------

    fpr: 1D array
        The false positive rates, calculated as false positives / total negatives, corresponding to each threshold in
        `thresholds`. Also equal to 1 - specificity.

    tpr: 1D array
        The true positive rates, calculated as true positives / total positives, corresponding to each threshold in
        `thresholds`. Also known as recall.

    thresholds: 1D array
        The probability thresholds at which either the false positive or true positive rates change.
    '''
    def points():
        '''
        Make an iterator over the points of the ROC curve and the thresholds generating them. See `roc_curve`.

        Yields
        ----------

        fpr: float
            The false positive rate corresponding to `threshold`.

        tpr: float
            `The true positive rate corresponding to `threshold`.

        threshold: float
            The probability threshold at which either the false positive or true positive rate changed.
        '''

        # TO-DO: refactor for null value check

        present = np.unique(y_true)
        if len(present) < 2:
            raise ValueError('The ROC curve is undefined when samples of only one class ({}) are available.'
                             .format(present[0]))

        y_pairs = np.stack((y_true, y_proba), axis=1)
        uniques = np.unique(y_pairs[:, 1])
        sorted_indices = np.argsort(uniques)
        sorted_uniques = uniques[sorted_indices]

        if 0.0 not in sorted_uniques:
            yield (1.0, 1.0, 0.0)

        # Perhaps this should start from threshold 1.0 and go backwards, since 1.0 corresponds with the origin. Also,
        # there are other ways to write the formula for FPR and TPR, but I think this is nicely symmetric and readable.

        for threshold in sorted_uniques:
            positives = (y_proba >= threshold).astype(np.int)
            fpr = ((positives == 1) & (y_true == 0)).sum() / (y_true == 0).sum()
            tpr = ((positives == 1) & (y_true == 1)).sum() / (y_true == 1).sum()
            yield (fpr, tpr, threshold)

        if 1.0 not in sorted_uniques:
            yield (0.0, 0.0, 1.0)

    roc_points = np.array(tuple(points()))
    return np.split(np.concatenate(roc_points.T), 3)


def roc_auc_score(y_true, y_proba):
    try:
        x, y, _ = roc_curve(y_true, y_proba)
        return auc(x, y)

    except ValueError:
        warn(RuntimeWarning('ROC-AUC score is undefined when there are no true positives. Returning np.nan.'))
        return np.nan


def auc(x, y):
    '''
    Calculate the signed area under a curve using the trapezoid rule.

    Parameters
    ----------

    x: 1D iterable
        The x-coordinates of the points on the curve.

    y: 1D iterable
        The y-coordinates of the points on the curve.

    Returns
    ----------

    auc: float
        The area under the curve.

    Notes
    ----------

    Areas below the x-axis will be treated as negative, and so the result of this function can be zero or negative.
    '''
    def area(x1, x2, y1, y2):
        return abs(x2 - x1) * (y1 + y2) / 2

    return sum(area(x1, x2, y1, y2) for x1, x2, y1, y2 in zip(x, x[1:], y, y[1:]))


def gini_impurity(y):
    """
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

.   Examples
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
    """
    p = np.unique(y, return_counts=True)[1] / y.size
    return 1 - sum(p * (1 - p))
