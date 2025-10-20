
import unittest
import numpy as np
from numpy.testing import assert_equal
from RADAR.time_series.algorithms import transformers
from RADAR.time_series.time_series_utils import TimeSeriesProcessor
from RADAR.metrics_module import metric_AUC_ROC
from sklearn.metrics import roc_auc_score



def generate_data(n_train=4000, n_test=1000, n_features=5, contamination=0.1, random_state=42):
    """
    Generates synthetic data for anomaly detection testing.
    
    Return:
        X_train: (n_train, n_features)
        X_test:  (n_test, n_features)
        y_train: (n_train,)
        y_test:  (n_test,)
    """
    rng = np.random.RandomState(random_state)
    
    # Datos normales
    X_train = rng.normal(0, 1, (n_train, n_features))
    X_test = rng.normal(0, 1, (n_test, n_features))
    
    # Etiquetas (0 = normal)
    y_train = np.zeros(n_train, dtype=int)
    y_test = np.zeros(n_test, dtype=int)
    
    # Introducir anomalías en el conjunto de test
    n_anom_train = int(contamination * n_train)
    n_anom_test = int(contamination * n_test)
    
    idx_train = rng.choice(n_train, n_anom_train, replace=False)
    idx_test = rng.choice(n_test, n_anom_test, replace=False)
    
    # Añadir desviación a las anomalías
    X_train[idx_train] += rng.normal(5, 1, (n_anom_train, n_features))
    X_test[idx_test] += rng.normal(5, 1, (n_anom_test, n_features))
    
    # Etiquetas de anomalías
    y_train[idx_train] = 1
    y_test[idx_test] = 1
    
    return X_train, X_test, y_train, y_test

class TestTransformer(unittest.TestCase):
    
    def setUp(self):
        self.n_train = 8000
        self.n_test = 2000
        self.contamination = 0.1
        self.n_features = 5
        self.seq_len = 24
        self.roc_floor = 0.7

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features= self.n_features,
            contamination=self.contamination, random_state=42
        )
        
        processor = TimeSeriesProcessor(window_size= self.seq_len, step_size=1, future_prediction=False)
        self.X_train_windows, self.y_train_windows, self.X_test_windows, self.y_test_windows = processor.process_train_test(self.X_train, self.y_train, self.X_test, self.y_test)
        print("X_train shape:", self.X_train_windows.shape)
        print("y_train shape:", self.y_train_windows.shape)
        print("X_test shape:", self.X_test_windows.shape)
        print("y_test shape:", self.y_test_windows.shape)

        kwargs ={
        "algorithm_": "transformer",
        "label_parser": None,
        "size_enc_in":self.n_features,
        "size_dec_in": self.n_features,
        "ulayers_feedfwd": 128,
        "seq_len":self.seq_len,
        "d_qk":64,
        "d_v":64,
        "d_model":64,
        "n_layers": 2,
        "n_heads": 8,
        "dropout_rate": 0.1,
        "attns_outs":False, 
        "train_epochs": 3,
        "batch_size": 16,
        "lr": 1e-3
         }
    

        self.modelTransformer = transformers.TransformersAnomalyDetection(**kwargs)
        self.modelTransformer.fit(self.X_train_windows)


    def test_prediction_scores(self):
        pred_scores = self.modelTransformer.decision_function(self.X_test_windows)
        print("Anomaly scores:", pred_scores.shape)
        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test_windows.shape[0])
        
        
        
    def test_prediction_labels(self):
        self.modelTransformer.evaluate(self.X_test_windows,self.y_test_windows)
        labels_pred = self.modelTransformer.labels_preds
        labels_true = np.array(self.y_test_windows).ravel()
        # check labels shapes
        assert_equal(labels_pred.shape, labels_true.shape)

class TestInformer(unittest.TestCase):
    
    def setUp(self):
        self.n_train = 8000
        self.n_test = 2000
        self.contamination = 0.1
        self.n_features = 5
        self.seq_len = 24
        self.roc_floor = 0.7

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features= self.n_features,
            contamination=self.contamination, random_state=42
        )
        
        processor = TimeSeriesProcessor(window_size= self.seq_len, step_size=1, future_prediction=False)
        self.X_train_windows, self.y_train_windows, self.X_test_windows, self.y_test_windows = processor.process_train_test(self.X_train, self.y_train, self.X_test, self.y_test)
        print("X_train shape:", self.X_train_windows.shape)
        print("y_train shape:", self.y_train_windows.shape)
        print("X_test shape:", self.X_test_windows.shape)
        print("y_test shape:", self.y_test_windows.shape)

        kwargs = {
        "algorithm_": "informer",
        "label_parser": None,
        "enc_in": self.n_features,             # Number of input variables for the encoder
        "dec_in": self.n_features,             # Number of input variables for the decoder
        "c_out": self.n_features,              # Output dimension (change if needed)
        "seq_len": self.seq_len,              # Input sequence length
        "label_len": self.seq_len,          # Length of the decoder input (label segment)
        "out_len": self.seq_len,             # Prediction length
        "factor": 5,                     # ProbSparse factor (used in ProbAttention)
        "d_model": 64,              # Model dimension
        "n_heads": 8,                    # Number of attention heads
        "e_layers": 2,                   # Number of encoder layers
        "d_layers": 1,                   # Number of decoder layers
        "d_ff": 128,                     # Feedforward network dimension
        "dropout": 0.1,                  # Dropout rate
        "attn": "prob",                  # Attention type: 'prob' or 'full'
        "activation": "gelu",           # Activation function
        "output_attention": False,       # Whether to output attention weights
        "distil": True,                  # Whether to use distillation in the encoder
        "mix": True,                     # Whether to use mixed attention in the decoder
        "train_epochs": 3,    # Number of training epochs
        "batch_size": 16,        # Batch size
        "lr": 1e-3                       # Learning rate
        }
    

        self.modelTransformer = transformers.TransformersAnomalyDetection(**kwargs)
        self.modelTransformer.fit(self.X_train_windows)


    def test_prediction_scores(self):
        pred_scores = self.modelTransformer.decision_function(self.X_test_windows)
        print("Anomaly scores:", pred_scores.shape)
        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test_windows.shape[0])
        
        
        
    def test_prediction_labels(self):
        self.modelTransformer.evaluate(self.X_test_windows,self.y_test_windows)
        labels_pred = self.modelTransformer.labels_preds
        labels_true = np.array(self.y_test_windows).ravel()
        # check labels shapes
        assert_equal(labels_pred.shape, labels_true.shape)


class TestAutoformer(unittest.TestCase):
    
    def setUp(self):
        self.n_train = 8000
        self.n_test = 2000
        self.contamination = 0.1
        self.n_features = 5
        self.seq_len = 24
        self.roc_floor = 0.7

        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test, n_features= self.n_features,
            contamination=self.contamination, random_state=42
        )
        
        processor = TimeSeriesProcessor(window_size= self.seq_len, step_size=1, future_prediction=False)
        self.X_train_windows, self.y_train_windows, self.X_test_windows, self.y_test_windows = processor.process_train_test(self.X_train, self.y_train, self.X_test, self.y_test)
        print("X_train shape:", self.X_train_windows.shape)
        print("y_train shape:", self.y_train_windows.shape)
        print("X_test shape:", self.X_test_windows.shape)
        print("y_test shape:", self.y_test_windows.shape)

        kwargs = {
        "algorithm_": "autoformer",
        "label_parser": None,
        "seq_len": self.seq_len,
        "label_len": self.seq_len,
        "pred_len": self.seq_len,
        "enc_in": self.n_features,
        "dec_in": self.n_features,
        "c_out": self.n_features,
        "d_model": 64,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 128,
        "moving_avg": 25,
        "factor": 5,
        "dropout": 0.1,
        "activation": "gelu",
        "output_attention": False,
        "train_epochs": 3,    # Number of training epochs
        "batch_size": 16,        # Batch size
        "lr": 1e-3                       # Learning rate
        }
    

        self.modelTransformer = transformers.TransformersAnomalyDetection(**kwargs)
        self.modelTransformer.fit(self.X_train_windows)


    def test_prediction_scores(self):
        pred_scores = self.modelTransformer.decision_function(self.X_test_windows)
        print("Anomaly scores:", pred_scores.shape)
        # check score shapes
        assert_equal(pred_scores.shape[0], self.X_test_windows.shape[0])
        
        
        
    def test_prediction_labels(self):
        self.modelTransformer.evaluate(self.X_test_windows,self.y_test_windows)
        labels_pred = self.modelTransformer.labels_preds
        labels_true = np.array(self.y_test_windows).ravel()
        # check labels shapes
        assert_equal(labels_pred.shape, labels_true.shape)






        
        




if __name__ == "__main__":
    unittest.main()