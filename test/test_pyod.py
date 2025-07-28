import unittest
from os import path
import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from SADL.static_data.algorithms import pyod
from pyod.utils.data import generate_data
from pyod.utils.data import generate_data_clusters
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

        kwargs = {"algorithm_": "abod","contamination":self.contamination,"method":'default'}

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

        kwargs = {"algorithm_": "cblof","contamination":self.contamination}

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
        
        kwargs = {"algorithm_": "alad","epochs":100, "latent_dim": 2, "learning_rate_disc":0.0001, "learning_rate_gen":0.0001, "dropout_rate":0.2, "add_recon_loss":False,
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


class TestAnoGAN(unittest.TestCase):
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

        
        kwargs = {"algorithm_": "anogan","epochs":3,
                "contamination":self.contamination}

        self.clf = pyod.PyodAnomalyDetection(**kwargs)
        self.clf.fit(self.X_train)

    """
    def test_prediction_scores(self):
        pred_scores = self.clf.decision_function(self.X_test)

        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (roc_auc_score(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        pred_labels = self.clf.predict(self.X_test)
        assert_equal(pred_labels.shape, self.y_test.shape)
    """

    def tearDown(self):
        pass


class TestFeatureBagging(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        kwargs = {"algorithm_": "feature_bagging","contamination":self.contamination}

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

class TestHBOS(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        kwargs = {"algorithm_": "hbos","contamination":self.contamination}

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

class TestIForest(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        kwargs = {"algorithm_": "iforest","contamination":self.contamination, "random_state": 42}

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

class TestKNN(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"algorithm_": "knn","contamination":self.contamination}

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

class TestLOF(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"algorithm_": "lof","contamination":self.contamination}

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

class TestMCD(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"algorithm_": "mcd","contamination":self.contamination, "random_state": 42}

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

class TestOCSVM(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"algorithm_": "ocsvm"}

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

class TestPCA(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"algorithm_": "pca", "contamination": self.contamination, "random_state": 42}

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


class TestLSCP(unittest.TestCase):
    def setUp(self):
        # Define data file and read X and y
        # Generate some data if the source data is missing
        this_directory = path.abspath(path.dirname(__file__))
        mat_file = 'cardio.mat'
        try:
            mat = loadmat(path.join(*[this_directory, 'data', mat_file]))

        except TypeError:
            print('{data_file} does not exist. Use generated data'.format(
                data_file=mat_file))
            X, y = generate_data(train_only=True)  # load data
        except IOError:
            print('{data_file} does not exist. Use generated data'.format(
                data_file=mat_file))
            X, y = generate_data(train_only=True)  # load data
        else:
            X = mat['X']
            y = mat['y'].ravel()
            X, y = check_X_y(X, y)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.4, random_state=42)

        kwargs_lof = {"algorithm_": "lof"}

        self.detector_list = [pyod.PyodAnomalyDetection(**kwargs_lof).model, pyod.PyodAnomalyDetection(**kwargs_lof).model]

        kwargs = {"algorithm_": "lscp", "detector_list": self.detector_list}

        self.clf = pyod.PyodAnomalyDetection(**kwargs)
        self.clf.fit(self.X_train)
        self.roc_floor = 0.6

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


class TestINNE(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"algorithm_": "inne", "contamination": self.contamination, "random_state": 42}

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

class TestGMM(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.n_components = 4
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data_clusters(
            n_train=self.n_train,
            n_test=self.n_test,
            n_clusters=self.n_components,
            contamination=self.contamination,
            random_state=42,
        )
        
        kwargs = {"algorithm_": "gmm", "n_components": self.n_components, "contamination": self.contamination}

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

class TestKDE(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"algorithm_": "kde", "contamination": self.contamination}

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


class TestLMDD(unittest.TestCase):
    def setUp(self):
        self.n_train = 100
        self.n_test = 50
        self.contamination = 0.1
        self.roc_floor = 0.6
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)
        
        kwargs = {"algorithm_": "lmdd", "contamination": self.contamination, "random_state": 42}

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