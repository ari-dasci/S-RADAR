import unittest
from os import path
import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from SADL.static_data.algorithms import sklearn
from pyod.utils.data import generate_data
from pyod.utils.data import generate_data_clusters
from numpy.testing import assert_equal

from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
    assert_almost_equal,
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


class TestSGDOneClassSVM(unittest.TestCase):
    def setUp(self):
        self.X = sp.csr_matrix(X)
        self.clf = sklearn.SkLearnAnomalyDetection(**{"algorithm_": "sgdocsvm"})

    def test_fit(self):
        return self.clf.fit(self.X)

    def test_decision_function(self):
        return self.clf.decision_function(self.X)


if __name__ == '__main__':
    unittest.main()