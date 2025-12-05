from RADAR.base_algorithm_module import BaseAnomalyDetection
from RADAR.metrics_module import print_metrics
from pyod.models.cblof import CBLOF
from pyod.models.abod import ABOD
from pyod.models.alad import ALAD
from pyod.models.anogan import AnoGAN
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.models.inne import INNE
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.lmdd import LMDD
import numpy as np

from inspect import signature

pyod_algorithms = {
    "cblof" : CBLOF,
    "abod" : ABOD,
    "alad" : ALAD,
    "anogan": AnoGAN,
    "feature_bagging": FeatureBagging,
    "hbos": HBOS,
    "iforest" : IForest,
    "knn" : KNN,
    "lof" : LOF,
    "mcd" : MCD,
    "ocsvm" : OCSVM,
    "pca" : PCA,
    "lscp" : LSCP,
    "inne" : INNE,
    "gmm" : GMM,
    "kde" : KDE,
    "lmdd" : LMDD,
}

class PyodAnomalyDetection(BaseAnomalyDetection):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.algorithm_ = pyod_algorithms[kwargs.get('algorithm_', 'abod')]# Default to ABOD

        self.model = None
        self.set_params(**kwargs)
        
        
      
    @classmethod
    def register_algorithm(cls, name, model_class):
        """Register a new algorithm in the class.
        Parameters:
          - name (str): The name of the new algorithm.
          - model_class (class): The class implementing the anomaly detection model.
        The class should have:
            - An __init__ method that accepts model-specific parameters.
            - A fit(X, y) method to train the model.
            - A predict(X) method to make predictions.
            - Optionally, a decision_function(X) for scoring anomalies.
        """
        if name in pyod_algorithms:
            print(
                f" The algorithm {name} is already registered and will be overwritten."
            )
        pyod_algorithms[name]  = model_class
        print(f"Algorithm {name} registered successfully.")
     
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        try:
            self.model.fit(X, y)
        except Exception as e:
            print("PYODerror fit():", str(e))
            print("For further reference please see: https://pyod.readthedocs.io/en/latest/")
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        try:
            return self.model.decision_function(X)
        except Exception as e:
            print("PYODerror decision_function():", str(e))
            print("For further reference please see: https://pyod.readthedocs.io/en/latest/")

    def predict(self, X):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.

        If label_parser is an attribute, then we execute the particular predict function

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        if "label_parser" in self.get_params().keys() and self.label_parser != None:
            return self.label_parser(X)
        else:
            try:
                return self.model.predict(X)
            except Exception as e:
                print("PYODerror predict():", str(e))
                print("For further reference please see: https://pyod.readthedocs.io/en/latest/")
    
    def evaluate(self, X, y=None):
        """
        Evaluates the model on data X (and optionally y).
        Uses decision_function to get anomaly scores and predict to get labels.
        If y is provided, prints metrics using print_metrics.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        y : numpy array of shape (n_samples,), optional
            Ground truth labels (0 for normal, 1 for anomaly).

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'scores': anomaly scores from decision_function
            - 'labels_preds': predicted labels (0 for normal, 1 for anomaly)
            - 'labels_true': ground truth labels (if y is provided)
        """
        try:
            # Get anomaly scores
            y_scores = self.decision_function(X)
            
            # Get predicted labels
            if "label_parser" in self.get_params().keys() and self.label_parser is not None:
                self.labels_preds = self.label_parser(y_scores)
            else:
                self.labels_preds = self.predict(X)
            
            # Ensure labels_preds is a numpy array
            if not isinstance(self.labels_preds, np.ndarray):
                self.labels_preds = np.array(self.labels_preds)
            
            # Ensure y_scores is a numpy array
            if not isinstance(y_scores, np.ndarray):
                y_scores = np.array(y_scores)
            
            results = {
                "scores": y_scores,
                "labels_preds": self.labels_preds,
            }
            
            # If ground truth is provided, calculate and print metrics
            if y is not None:
                y_true = np.array(y).flatten()
                print_metrics(["Accuracy", "F1", "Recall", "Precision"], y_true, self.labels_preds)
                results["labels_true"] = y_true
            
            return results
            
        except Exception as e:
            print("PYODerror evaluate():", str(e))
            print("For further reference please see: https://pyod.readthedocs.io/en/latest/")
            raise

    def set_params(self, **params): #Este setea sus propios parametros
        """Set the parameters of this estimator.
        Returns
        -------
        self : object
        """
        
        super().set_params(**params) #Llama al base para setear sus parametros en caso de que los hubiera

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        
        self.algorithm_ = pyod_algorithms[params.get('algorithm_', 'abod')]# Default to ABOD
        
        valid_params = self.get_default_params(**params)
        
        setattr(self.algorithm_,"algorithm_",valid_params["algorithm_"])
   
        #Setear el modelo particular
        model_error = False
        for key, value in params.items():
            if key != "algorithm_" and key != "label_parser": #TODO: se puede hacer un get_local_params para ver los parametros de la clase padre para que se los salte
                if key not in valid_params: #Si hay alguna variable no aceptada por el modelo 
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator {self}.{self.algorithm_} "
                        f"Valid parameters are: {valid_params!r}."
                    )

        # Set positional parameters specific to the model
        positional_params = {}
        init_signature = signature(self.algorithm_.__init__)
        for param_name, param in init_signature.parameters.items():
            if param_name != 'self' and param.default == param.empty:  # Check for positional parameters
                if param_name in params.keys():
                    positional_params[param_name] = params[param_name]

        if not model_error:
            try:
                self.model = self.algorithm_(**positional_params)
            except Exception as e:
                print("PYODerror: ", str(e))
                print("For further reference please see: https://pyod.readthedocs.io/en/latest/")
                raise

            for key, value in params.items():
                setattr(self.model, key, value)

        return self
    

    def get_default_params(self, **params): #Get default params based on the positional params already given
        """Get DEFAULT parameters for this estimator, params is used to configure positional parameters in order to
        obtain default parameters of the object.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        out = super().get_params()
        
        init = getattr(self.algorithm_.__init__, 'deprecated_original', self.algorithm_.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            param_names = []
        # intrpect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(self.algorithm_.__init__)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (self.algorithm_, init_signature))
        # Extract and sort argument names excluding 'self'
        param_names = sorted([p.name for p in parameters])
        
        # Set positional parameters specific to the model
        positional_params = {}
        init_signature = signature(self.algorithm_.__init__)
        
        for param_name, param in init_signature.parameters.items():
            if param_name != 'self' and param.default == param.empty:  # Check for positional parameters
                if param_name in params.keys():
                    positional_params[param_name] = params[param_name]

        try:
            nuevo_modelo = self.algorithm_(**positional_params)
        except Exception as e:
            print("PYODerror: ", str(e))
            print("For further reference please see: https://pyod.readthedocs.io/en/latest/")
            raise

        out["algorithm_"] = self.algorithm_.__name__
        for key in param_names:
            if hasattr(self.model, key):
                out[key] = getattr(self.model, key)
            else:
                out[key] = getattr(nuevo_modelo, key, None) #Default value or None value

        return out
    

    def get_params(self):
        """Get parameters for this estimator.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        out = dict()
        out = super().get_params()
        out["algorithm_"] = self.algorithm_.__name__

        try:
            param_names = self.model.get_params()
        except Exception as e:
            print("PYODerror get_params():", str(e))
            print("For further reference please see: https://pyod.readthedocs.io/en/latest/")
            
        for key in param_names:
            if hasattr(self.model, key):
                out[key] = getattr(self.model, key)

        return out

    