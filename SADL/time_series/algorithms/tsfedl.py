from SADL.base_algorithm_module import BaseAnomalyDetection
import pytorch_lightning as pl
import numpy as np
import traceback
from torch.nn import functional as F
import torch
from inspect import signature
from TSFEDL.models_pytorch import OhShuLih_Classifier
from TSFEDL.models_pytorch import OhShuLih

tsfedl_algorithms = {
    "ohshulih_classifier" : OhShuLih_Classifier,
    "ohshulih" : OhShuLih,
}

class TsfedlAnomalyDetection(BaseAnomalyDetection):
    """
    Base module for any pyTorch Lightning based algorithm in TSFEDL library

    Parameters
    ----------
        (super) label_parser: function of shape (n_samples,) with the 
        specific methods or operations to apply to the score values.

        algorithm_: class object of the specific model

        model: object containing the specific model. To see the particular attributes of each model see: https://s-tsfe-dl.readthedocs.io/en/latest/index.html
        
        pytorch_params_: dict object of pl params for the Trainer object.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.algorithm_ = tsfedl_algorithms[kwargs.get('algorithm_', 'ohshulih_classifier')]# Default to OhShuLih Classifier

        self.model = None
        self.pytorch_params_ = {}
        self.set_params(**kwargs)

    def fit(self, X, y=None):
        try:
            if("pytorch_params_" in self.get_params().keys()):
                trainer = pl.Trainer(**self.pytorch_params_)
                trainer.fit(self.model, X)
            else: 
                pl.Trainer().fit(self.model, X)
        except Exception as e:
            print("TSFEDLerror fit(): ", str(e))
            print("For further reference please see: https://s-tsfe-dl.readthedocs.io/en/latest/index.html")
            raise
        return self
    
    def decision_function(self, X):
        decision_scores_list = []
        for data, labels in X:
            print(data.shape)
            decision_scores = self.model(data)
            print(decision_scores.shape)
            #X_pred = decision_scores.detach().numpy()
            #X = data.detach().numpy()
            #X = np.repeat(X, 1000, axis=0).reshape(1, 1, -1)
            #sum_of_norms = np.sum(np.linalg.norm(X_pred - X, axis=2))

            #decision_scores_list.append(sum_of_norms)
        
        #return decision_scores_list
        #return np.sum(np.linalg.norm(X_pred-X), axis = 1)

    def predict(self, X):
        if "label_parser" in self.get_params().keys() and self.label_parser != None:
            return self.label_parser(X)
        else:
            print("TSFEDLerror predict(): no label_parser function set to make a prediction, please provide one.")


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
        
        self.algorithm_ = tsfedl_algorithms[params.get('algorithm_', 'ohshulih_classifier')]# Default to OhShuLih Classifier
        
        valid_params = self.get_default_params(**params)
        
        setattr(self.algorithm_,"algorithm_",valid_params["algorithm_"])
   

        #Set specific pytorch params 
        pytorch_params = {}
        pytorch_signature = signature(pl.Trainer.__init__)
        for param_name, param in pytorch_signature.parameters.items():
            if param_name in params.keys():
                pytorch_params[param_name] = params[param_name]            
                del params[param_name]

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
                print("TSFEDLerror:", str(e))
                print("For further reference please see: https://s-tsfe-dl.readthedocs.io/en/latest/index.html")
                raise

            for key, value in params.items():
                setattr(self.model, key, value)
            for key, value in pytorch_params.items():
                self.pytorch_params_[key] = value

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
            print("TSFEDLerror:", str(e))
            print("For further reference please see: https://s-tsfe-dl.readthedocs.io/en/latest/index.html")
            raise

        out["algorithm_"] = self.algorithm_.__name__
        out["pytorch_params_"] = {}
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

        init_signature = signature(self.algorithm_.__init__)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        param_names = sorted([p.name for p in parameters])
        for key in param_names:
            if hasattr(self.model, key):
                out[key] = getattr(self.model, key)
            else:
                out[key] = getattr(self.model, key, None)

        out["pytorch_params_"] = self.pytorch_params_
        return out