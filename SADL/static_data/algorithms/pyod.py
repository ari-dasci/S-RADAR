import sys 
import os
import pandas
from SADL.base_algorithm_module import BaseAnomalyDetection
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


from inspect import signature
from collections import defaultdict

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

        if "label_parser" in self.get_params().keys() and self.label_parser != None:
            return self.label_parser(X)
        else:
            try:
                return self.model.predict(X)
            except Exception as e:
                print("PYODerror predict():", str(e))
                print("For further reference please see: https://pyod.readthedocs.io/en/latest/")
            

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

        valid_params = self.get_params()
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
                positional_params[param_name] = params[param_name]

        if not model_error:
            self.model = self.algorithm_(**positional_params)
            for key, value in params.items():
                setattr(self.model, key, value)

        return self

    def get_params(self):
        """Get parameters for this estimator.

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
                positional_params[param_name] = [LOF(),LOF()] #TODO arreglar esto


        nuevo_modelo = self.algorithm_(**positional_params)
        out["algorithm_"] = self.algorithm_.__name__
        for key in param_names:
            if hasattr(self.model, key):
                out[key] = getattr(self.model, key)
            else:
                out[key] = getattr(nuevo_modelo, key, None) #Default value or None value
        print(out)
        return out

    