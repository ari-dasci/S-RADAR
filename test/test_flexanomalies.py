import numpy as np
import unittest
from pyod.utils.data import generate_data
from numpy.testing import assert_equal
from RADAR.metrics_module import metric_AUC_ROC
from RADAR.federated_data.algorithms import flexanomalies
from RADAR.static_data.static_datasets_uci import global_load
from sklearn.model_selection import train_test_split
from pyod.utils.data import generate_data

class TestIforest(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(42)
        self.contamination = 0.1
        self.n_train = 200
        self.n_test = 50
        self.roc_floor = 0.8

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)
        
        # X,y = global_load('default_of_credit_card_clients')   #name dataset in static datasets uci repo
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42) #Agregar un generate dataset a datasets
        kwargs = {
            "algorithm_": "isolationForest",
            "contamination":self.contamination,
            "label_parser": None,
            "n_estimators": 100, 
            "n_rounds": 5,
            "n_clients":3,          
        }   
        self.modelIforest = flexanomalies.FlexAnomalyDetection(**kwargs)
        self.modelIforest.fit(self.X_train,self.y_train)

    def test_prediction_scores(self):
        pred_scores = self.modelIforest.decision_function(self.X_test)
        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])

        # check performance
        assert (metric_AUC_ROC(self.y_test, pred_scores) >= self.roc_floor)

    def test_prediction_labels(self):
        self.modelIforest.predict(self.X_test)
        pred_labels = self.modelIforest.model.labels_
        assert_equal(pred_labels.shape, self.y_test.shape)
    
     
    
        

class TestAEncoder(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(42)
        self.contamination = 0.1
        self.n_train = 200
        self.n_test = 50
        self.roc_floor = 0.8

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)
            # X,y = global_load('default_of_credit_card_clients')   #name dataset in static datasets uci repo
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42) #Agregar un generate dataset a datasets
        kwargs = {
        "algorithm_": "autoencoder",
        "contamination":0.1,
        "label_parser": None,
        "epochs": 100,
        "input_dim": self.X_train.shape[1],
        "batch_size": 8,
        "neurons": [16,8, 16],
        "hidden_act": ['relu', 'relu', 'relu'],
        "n_clients":2,
        "n_rounds":10,
        }  
        self.modelAE = flexanomalies.FlexAnomalyDetection(**kwargs)
        self.modelAE.fit(self.X_train,self.y_train)

    def test_prediction_scores(self):
        pred_scores = self.modelAE.predict(self.X_test)
        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test.shape[0])
        # check performance
        assert (metric_AUC_ROC(self.y_test, self.modelAE.model.labels_) >= self.roc_floor)
        
        
    def test_prediction_labels(self):
        self.modelAE.predict(self.X_test)
        pred_labels = self.modelAE.model.labels_
        assert_equal(pred_labels.shape, self.y_test.shape)
        
























        
if __name__ == '__main__':
    unittest.main()
    
    