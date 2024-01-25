from ...base_algorithm_module import BaseAnomalyDetection
from pyod.models.cblof import CBLOF
from pyod.models.abod import ABOD
from inspect import signature
from ...base_utils_module import check
import config_pyod #TODO: preguntar a nacho si esto esta bien
from collections import defaultdict

pyod_algorithms = {
    "cblof" : CBLOF,
    "abod" : ABOD,
}

class PyodAnomalyDetection(BaseAnomalyDetection):

    def __init__(self, **kwargs):
        algorithm = kwargs.get('algorithm', 'abod')  # Default to ABOD
        
        if algorithm in pyod_algorithms:
            selected_class = pyod_algorithms[algorithm]
        else:
            selected_class = ABOD

        self.algorithm = selected_class
        self.model = None
        self.set_params(**kwargs)

    def fit(self, X, y=None):
        self.model.fit(X, y)

    def decision_function(self, X):
        self.model.decision_function(X)

    def predict(self, X):

        #if "label_parser" in self.key
        X = self.label_parser(X)
        self.model.predict(X)

    def set_params(self, **params): #Este setea sus propios parametros
        super().set_params(**params) #Llama al base para setear sus parametros en caso de que los hubiera

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        
        valid_params = self.get_params()
        valid_params["algorithm"] = params["algorithm"]
        setattr(self.algorithm,"algorithm",valid_params["algorithm"])

        #Setear el modelo particular
        model_error = False
        for key, value in params.items():
            if key != "algorithm" and key != "label_parser": #TODO: se puede hacer un get_local_params para ver los parametros de la clase padre para que se los salte
                if key not in valid_params: #Si hay alguna variable no aceptada por el modelo 
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator {self}.{self.algorithm} "
                        f"Valid parameters are: {valid_params!r}."
                    )
                else:
                    #Load config parameters for the particular algorithm
                    config_pyod['PYOD_PARAMETERS'][self.algorithm.__class__.__name__]
                    #if not check() de la variable es adecuado
                    #Solo print del error, el valor de la variable se setea al default
                    #Si hay parametro por defecto se pone, si no, no se podría instanciar el modelo
                    #params[key] = valid_params[key]
                    
        if not model_error:
            self.model = self.algorithm()
            for key, value in params.items():
                setattr(self.model, key, value)

        return self

    def get_params(self):
        out = dict()
        
        init = getattr(self.algorithm.__init__, 'deprecated_original', self.algorithm.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            param_names = []
        # intrpect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(self.algorithm.__init__)
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
                                   % (self.algorithm, init_signature))
        # Extract and sort argument names excluding 'self'
        param_names = sorted([p.name for p in parameters])
        
        nuevo_modelo = self.algorithm()
        out["algorithm"] = None
        for key in param_names:
            if hasattr(self.model, key):
                out[key] = getattr(self.model, key)
            else:
                out[key] = getattr(nuevo_modelo, key)

        return out

    