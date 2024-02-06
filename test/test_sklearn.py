import unittest
from os import path
import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import check_X_y
from sklearn.base import clone, is_classifier
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from SADL.static_data.algorithms import sklearn
from pyod.utils.data import generate_data
from pyod.utils.data import generate_data_clusters
from numpy.testing import assert_equal

from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
    assert_almost_equal,
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
"""
Unittest class following pyod test cases to try at once all the pyod models following our indications
"""

class TestElliptic(unittest.TestCase):

    def test_elliptic_envelope(self):
        rnd = np.random.RandomState(42)
        X = rnd.randn(100, 10)
        kwargs = {"algorithm_": "elliptic","contamination":0.1}
        clf = sklearn.SkLearnAnomalyDetection(**kwargs)
        clf.fit(X)
        y_pred = clf.predict(X)
        scores = clf.model.score_samples(X)
        decisions = clf.decision_function(X)

        assert_array_almost_equal(scores, -clf.model.mahalanobis(X))
        assert_array_almost_equal(clf.model.mahalanobis(X), clf.model.dist_)
        assert_almost_equal(
            clf.model.score(X, np.ones(100)), (100 - y_pred[y_pred == -1].size) / 100.0
        )
        assert sum(y_pred == -1) == sum(decisions < 0)

    
    def test_score_samples(self):
        X_train = [[1, 1], [1, 2], [2, 1]]
        kwargs = {"algorithm_": "elliptic","contamination":0.2}
        clf1 = sklearn.SkLearnAnomalyDetection(**kwargs).fit(X_train)
        clf2 = sklearn.SkLearnAnomalyDetection(**{"algorithm_": "elliptic"}).fit(X_train)
        assert_array_equal(
            clf1.model.score_samples([[2.0, 2.0]]),
            clf1.decision_function([[2.0, 2.0]]) + clf1.model.offset_,
        )
        assert_array_equal(
            clf2.model.score_samples([[2.0, 2.0]]),
            clf2.decision_function([[2.0, 2.0]]) + clf2.model.offset_,
        )
        assert_array_equal(
            clf1.model.score_samples([[2.0, 2.0]]), clf2.model.score_samples([[2.0, 2.0]])
        )




# Test Data

# test sample 1
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y = [1, 1, 1, 2, 2, 2]
T = np.array([[-1, -1], [2, 2], [3, 2]])
true_result = [1, 2, 2]

# test sample 2; string class labels
X2 = np.array(
    [
        [-1, 1],
        [-0.75, 0.5],
        [-1.5, 1.5],
        [1, 1],
        [0.75, 0.5],
        [1.5, 1.5],
        [-1, -1],
        [0, -0.5],
        [1, -1],
    ]
)
Y2 = ["one"] * 3 + ["two"] * 3 + ["three"] * 3
T2 = np.array([[-1.5, 0.5], [1, 2], [0, -2]])
true_result2 = ["one", "two", "three"]

# test sample 3
X3 = np.array(
    [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)
Y3 = np.array([1, 1, 1, 1, 2, 2, 2, 2])

# test sample 4 - two more or less redundant feature groups
X4 = np.array(
    [
        [1, 0.9, 0.8, 0, 0, 0],
        [1, 0.84, 0.98, 0, 0, 0],
        [1, 0.96, 0.88, 0, 0, 0],
        [1, 0.91, 0.99, 0, 0, 0],
        [0, 0, 0, 0.89, 0.91, 1],
        [0, 0, 0, 0.79, 0.84, 1],
        [0, 0, 0, 0.91, 0.95, 1],
        [0, 0, 0, 0.93, 1, 1],
    ]
)
Y4 = np.array([1, 1, 1, 1, 2, 2, 2, 2])

# test sample 5 - test sample 1 as binary classification problem
X5 = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
Y5 = [1, 1, 1, 2, 2, 2]
true_result5 = [0, 1, 1]


class TestSGDOneClassSVM(unittest.TestCase):
    def setUp(self):
        self.X = sp.csr_matrix(X)
        self.clf = sklearn.SkLearnAnomalyDetection(**{"algorithm_": "sgdocsvm"})

    def test_fit(self):
        return self.clf.fit(self.X)

    def test_decision_function(self):
        return self.clf.decision_function(self.X)
    
   # def test_late_onset_averaging_not_reached(self):
   #     clf1 = sklearn.SkLearnAnomalyDetection(**{"algorithm_": "sgdocsvm", "average": 600, "tol": None}).model
   #     clf2 = sklearn.SkLearnAnomalyDetection(**{"algorithm_": "sgdocsvm", "tol": None}).model
   #     for _ in range(100):
   #         if is_classifier(clf1):
   #             clf1.partial_fit(X, Y, classes=np.unique(Y))
   #             clf2.partial_fit(X, Y, classes=np.unique(Y))
   #         else:
   #             clf1.partial_fit(X, Y)
   #             clf2.partial_fit(X, Y)

   #     assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=16)
   #     assert_allclose(clf1.offset_, clf2.offset_)



if __name__ == '__main__':
    unittest.main()