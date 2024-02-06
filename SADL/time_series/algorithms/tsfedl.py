from SADL.base_algorithm_module import BaseAnomalyDetection
import keras
from inspect import signature
from TSFEDL.models_keras import OhShuLih

tsfedl_algorithms = {
    "ohshulih" : OhShuLih,
}

class TsfedlAnomalyDetection(BaseAnomalyDetection):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.algorithm_ = tsfedl_algorithms[kwargs.get('algorithm_', 'ohshulih')]# Default to OhShuLih

        self.model = None
        self.set_params(**kwargs)

    def fit(self, X, y=None):
        pass
    
    def decision_function(self, X):
        #Specific implementation
        pass

    def predict(self, X):
        pass


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
        
        self.algorithm_ = tsfedl_algorithms[params.get('algorithm_', 'ohshulih')]# Default to ABOD
        
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