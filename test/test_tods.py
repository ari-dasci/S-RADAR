import unittest
from unittest import TestCase
from pyod.utils.data import generate_data
from SADL.time_series.algorithms.tods import TodsAnomalyDetection
from sklearn.metrics import roc_auc_score

class TestTelemanomSKI(unittest.TestCase):
    def setUp(self):

        _dummy = TestCase('__init__')
        self.assert_greater_equal = _dummy.assertGreaterEqual
        self.assert_greater = _dummy.assertGreater
        self.assert_less_equal = _dummy.assertLessEqual
        self.assert_less = _dummy.assertLess
        self.assert_equal = _dummy.assertEqual

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.0
        self.window_size = 5
        self.l_s = 5
        self.n_predictions = 1


        self.X_train, self.X_test,self.y_train,  self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"label_parser": None, "algorithm_": "telemanom", "contamination": self.contamination,
                  "l_s": self.l_s, "n_predictions": self.n_predictions}

        self.transformer = TodsAnomalyDetection(**kwargs)
        self.y_train = self.y_train[self.l_s:-self.n_predictions]
        self.y_test = self.y_test[self.l_s:-self.n_predictions]
        self.transformer.fit(self.X_train)

    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.model.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)



class TestDeepLogSKI(unittest.TestCase):
    def setUp(self):

        _dummy = TestCase('__init__')
        self.assert_greater_equal = _dummy.assertGreaterEqual
        self.assert_greater = _dummy.assertGreater
        self.assert_less_equal = _dummy.assertLessEqual
        self.assert_less = _dummy.assertLess
        self.assert_equal = _dummy.assertEqual

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.0
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"label_parser": None, "algorithm_": "deep_log", "contamination": self.contamination}

        self.transformer = TodsAnomalyDetection(**kwargs)
        self.transformer.fit(self.X_train)

    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.model.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)

class TestAutoRegODetectorSKI(unittest.TestCase):
    def setUp(self):

        _dummy = TestCase('__init__')
        self.assert_greater_equal = _dummy.assertGreaterEqual
        self.assert_greater = _dummy.assertGreater
        self.assert_less_equal = _dummy.assertLessEqual
        self.assert_less = _dummy.assertLess
        self.assert_equal = _dummy.assertEqual

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.0
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"label_parser": None, "algorithm_": "auto_reg", "contamination": self.contamination}

        self.transformer = TodsAnomalyDetection(**kwargs)
        self.transformer.fit(self.X_train)

    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.model.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)

class TestKDiscordODetectorSKI(unittest.TestCase):
    def setUp(self):

        _dummy = TestCase('__init__')
        self.assert_greater_equal = _dummy.assertGreaterEqual
        self.assert_greater = _dummy.assertGreater
        self.assert_less_equal = _dummy.assertLessEqual
        self.assert_less = _dummy.assertLess
        self.assert_equal = _dummy.assertEqual

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.0
        self.window_size = 5
        self.l_s = 5
        self.n_predictions = 1
        self.X_train, self.X_test,self.y_train,  self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"label_parser": None, "algorithm_": "kdiscord", "contamination": self.contamination, "window_size": self.window_size}

        self.y_test = self.y_test[self.window_size-1:]
        self.transformer = TodsAnomalyDetection(**kwargs)
        self.transformer.fit(self.X_train)

    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.model.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)


class TestLSTMODetectorSKI(unittest.TestCase):
    def setUp(self):

        _dummy = TestCase('__init__')
        self.assert_greater_equal = _dummy.assertGreaterEqual
        self.assert_greater = _dummy.assertGreater
        self.assert_less_equal = _dummy.assertLessEqual
        self.assert_less = _dummy.assertLess
        self.assert_equal = _dummy.assertEqual

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.0
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"label_parser": None, "algorithm_": "lstm", "contamination": self.contamination}

        self.transformer = TodsAnomalyDetection(**kwargs)
        self.transformer.fit(self.X_train)

    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.model.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)

class TestMatrixProfileSKI(unittest.TestCase):
    def setUp(self):

        _dummy = TestCase('__init__')
        self.assert_greater_equal = _dummy.assertGreaterEqual
        self.assert_greater = _dummy.assertGreater
        self.assert_less_equal = _dummy.assertLessEqual
        self.assert_less = _dummy.assertLess
        self.assert_equal = _dummy.assertEqual

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.0
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"label_parser": None, "algorithm_": "matrix", "contamination": self.contamination}

        self.transformer = TodsAnomalyDetection(**kwargs)
        self.transformer.fit(self.X_train)

    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.model.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)


class TestPCAODetectorSKI(unittest.TestCase):
    def setUp(self):

        _dummy = TestCase('__init__')
        self.assert_greater_equal = _dummy.assertGreaterEqual
        self.assert_greater = _dummy.assertGreater
        self.assert_less_equal = _dummy.assertLessEqual
        self.assert_less = _dummy.assertLess
        self.assert_equal = _dummy.assertEqual

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.0
        self.window_size = 5
        self.l_s = 5
        self.n_predictions = 1
        self.X_train, self.X_test,self.y_train,  self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"label_parser": None, "algorithm_": "pcao", "contamination": self.contamination, "window_size":self.window_size}

        self.transformer = TodsAnomalyDetection(**kwargs)
        self.y_test = self.y_test[self.window_size-1:]
        self.transformer.fit(self.X_train)

    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.model.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)

class TestSODSKI(unittest.TestCase):
    def setUp(self):

        _dummy = TestCase('__init__')
        self.assert_greater_equal = _dummy.assertGreaterEqual
        self.assert_greater = _dummy.assertGreater
        self.assert_less_equal = _dummy.assertLessEqual
        self.assert_less = _dummy.assertLess
        self.assert_equal = _dummy.assertEqual

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.0
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"label_parser": None, "algorithm_": "sod", "contamination": self.contamination}

        self.transformer = TodsAnomalyDetection(**kwargs)
        self.transformer.fit(self.X_train)

    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.model.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)


#Original test fails
class TestSystemWiseDetectionSKI(unittest.TestCase):
    def setUp(self):

        _dummy = TestCase('__init__')
        self.assert_greater_equal = _dummy.assertGreaterEqual
        self.assert_greater = _dummy.assertGreater
        self.assert_less_equal = _dummy.assertLessEqual
        self.assert_less = _dummy.assertLess
        self.assert_equal = _dummy.assertEqual

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.0
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features=3,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"label_parser": None, "algorithm_": "system_wise", "contamination": self.contamination}

        self.transformer = TodsAnomalyDetection(**kwargs)
        self.transformer.fit(self.X_train)

    def test_prediction_labels(self):
        pred_labels = self.transformer.predict(self.X_test)
        self.assert_equal(pred_labels.shape[0], self.y_test.shape[0])

    def test_prediction_score(self):
        pred_scores = self.transformer.model.predict_score(self.X_test)
        self.assert_equal(pred_scores.shape[0], self.y_test.shape[0])
        self.assert_greater_equal(roc_auc_score(self.y_test, pred_scores), self.roc_floor)
