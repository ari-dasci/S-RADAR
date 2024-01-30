import unittest
import os
import sys
import numpy as np
from SADL.static_data.algorithms import pyod
from pyod.utils.data import generate_data
from numpy.testing import assert_equal
from sklearn.metrics import roc_auc_score

"""
Unittest class following pyod test cases to try at once all the pyod models following our indications
"""

class TestABOD(unittest.TestCase):
    def setUp(self):
        self.n_train = 50
        self.n_test = 50
        self.contamination = 0.2
        self.roc_floor = 0.8

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        kwargs = {"algorithm": "abod","contamination":self.contamination,"method":'default'}

        self.clf = pyod.PyodAnomalyDetection(**kwargs)
        self.clf.fit(self.X_train)

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def tearDown(self):
        pass



class TestCBLOF(unittest.TestCase):
    def setUp(self):
        random_state = np.random.RandomState(42)
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        kwargs = {"algorithm": "cblof","contamination":self.contamination}

        self.clf = pyod.PyodAnomalyDetection(**kwargs)
        self.clf.fit(self.X_train)

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def tearDown(self):
        pass



class TestALAD(unittest.TestCase):
    def setUp(self):
        self.n_train = 500
        self.n_test = 200
        self.n_features = 2
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            n_features=self.n_features, contamination=self.contamination,
            random_state=42)
        
        kwargs = {"algorithm": "alad","epochs":100, "last_dim": 2, "learning_rate_disc":0.0001, "learning_rate_gen":0.0001, "dropout_rate":0.2, "add_recon_loss":False,
                        "lambda_recon_loss":0.05,
                        "add_disc_zz_loss":True,
                        "dec_layers":[75, 100],
                        "enc_layers":[100, 75],
                        "disc_xx_layers":[100, 75],
                        "disc_zz_layers":[25, 25],
                        "disc_xz_layers":[100, 75],
                        "spectral_normalization":False,
                        "activation_hidden_disc":'tanh',
                        "activation_hidden_gen":'tanh',
                        "preprocessing":True, "batch_size":200,
                        "contamination":self.contamination}

        self.clf = pyod.PyodAnomalyDetection(**kwargs)
        self.clf.fit(self.X_train)

    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)

    def tearDown(self):
        pass



if __name__ == '__main__':
    unittest.main()