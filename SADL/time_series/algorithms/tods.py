from inspect import signature
from SADL.base_algorithm_module import BaseAnomalyDetection
from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from tods.detection_algorithm.DeepLog import DeepLogPrimitive
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
from tods.detection_algorithm.Telemanom import TelemanomPrimitive
from tods.sk_interface.detection_algorithm.AutoRegODetector_skinterface import AutoRegODetectorSKI
from tods.detection_algorithm.AutoRegODetect import AutoRegODetectorPrimitive
from tods.sk_interface.detection_algorithm.KDiscordODetector_skinterface import KDiscordODetectorSKI
from tods.detection_algorithm.KDiscordODetect import KDiscordODetectorPrimitive
from tods.sk_interface.detection_algorithm.LSTMODetector_skinterface import LSTMODetectorSKI
from tods.detection_algorithm.LSTMODetect import LSTMODetectorPrimitive
from tods.sk_interface.detection_algorithm.MatrixProfile_skinterface import MatrixProfileSKI
from tods.detection_algorithm.MatrixProfile import MatrixProfilePrimitive
from tods.sk_interface.detection_algorithm.PCAODetector_skinterface import PCAODetectorSKI
from tods.detection_algorithm.PCAODetect import PCAODetectorPrimitive
from tods.sk_interface.detection_algorithm.SOD_skinterface import SODSKI
from tods.detection_algorithm.PyodSOD import SODPrimitive
from tods.sk_interface.detection_algorithm.SystemWiseDetection_skinterface import SystemWiseDetectionSKI
from tods.detection_algorithm.SystemWiseDetection import SystemWiseDetectionPrimitive

tods_algorithms = {
    "deep_log" : DeepLogSKI,
    "telemanom" : TelemanomSKI,
    "auto_reg" : AutoRegODetectorSKI,
    "kdiscord" : KDiscordODetectorSKI,
    "lstm" : LSTMODetectorSKI,
    "matrix" : MatrixProfileSKI,
    "pcao" : PCAODetectorSKI,
    "sod" : SODSKI, 
    "system_wise": SystemWiseDetectionSKI

}

corresponding_primitive = {
    "DeepLogSKI" : DeepLogPrimitive,
    "TelemanomSKI" : TelemanomPrimitive,
    "AutoRegODetectorSKI" : AutoRegODetectorPrimitive,
    "KDiscordODetectorSKI": KDiscordODetectorPrimitive,
    "LSTMODetectorSKI" : LSTMODetectorPrimitive,
    "MatrixProfileSKI" : MatrixProfilePrimitive,
    "PCAODetectorSKI" : PCAODetectorPrimitive,
    "SODSKI" : SODPrimitive,
    "SystemWiseDetectionSKI" : SystemWiseDetectionPrimitive
}

class TodsAnomalyDetection(BaseAnomalyDetection):
    """
    Base module for any pyTorch Lightning based algorithm in TODS library

    Parameters
    ----------
        (super) label_parser: function of shape (n_samples,) with the 
        specific methods or operations to apply to the score values.

        algorithm_: class object of the specific model

        model: object containing the specific model. To see the particular attributes of each model see: https://s-tsfe-dl.readthedocs.io/en/latest/index.html
        
        base_params_: 

        model_params_:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.algorithm_ = tods_algorithms[kwargs.get('algorithm_', 'deep_log')] #Default to DeepLogSKI 

        self.model = None
        self.primitive_params_= {}
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
            self.model.fit(X)
        except Exception as e:
            print("TODSError fit():", str(e))
            print("For further reference please see: https://tods-doc.github.io/index.html")
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. .

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
            return self.model.clf_.decision_function(X)
        except Exception as e:
            print("TODSError decision_function():", str(e))
            print("For further reference please see: https://tods-doc.github.io/index.html")
        

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
                print("TODSError predict():", str(e))
                print("For further reference please see: https://tods-doc.github.io/index.html")
            

    def set_params(self, **params):
        """Set the parameters of this estimator.
        Returns
        -------
        self : object
        """
        super().set_params(**params) #Llama al base para setear sus parametros en caso de que los hubiera

        if not params: 
            #Simple optimization to gain speed
            return self
        
        self.algorithm_ = tods_algorithms[params.get('algorithm_', 'deep_log')]# Default to DeepLog

        valid_params = self.get_default_params(**params)
        #print(valid_params)
        setattr(self.algorithm_,"algorithm_",valid_params["algorithm_"])

        #Set specific primitive params 
        primitive_params = {}
        primitive_signature = self.algorithm_().primitives[0].hyperparams
        for param_name in primitive_signature:
            if param_name in params.keys():
                primitive_params[param_name] = params[param_name]
                del params[param_name]

        #Setear el modelo particular
        for key, value in params.items():
            if key != "algorithm_" and key != "label_parser": #TODO: se puede hacer un get_local_params para ver los parametros de la clase padre para que se los salte
                #Check special case on Hyperparams object
                if key not in valid_params and key not in valid_params['primitive_params_'].keys(): #Si hay alguna variable no aceptada por el modelo 
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator {self}.{self.algorithm_} "
                        f"Valid parameters are: {valid_params!r}."
                    )
        try:
            self.model = self.algorithm_(**primitive_params)
        except Exception as e:
            print("TODSError:", str(e))
            print("For further reference please see: https://tods-doc.github.io/index.html")
            raise
        
        for key, value in params.items():
            setattr(self.model, key, value)
        self.primitive_params_ = self.model.primitives[0].hyperparams
        
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

        try:
            nuevo_modelo = self.algorithm_()
        except Exception as e:
            print("TODSError:", str(e))
            print("For further reference please see: https://tods-doc.github.io/index.html")
            raise

        out["algorithm_"] = self.algorithm_.__name__
        out["primitive_params_"] = self.algorithm_().primitives[0].hyperparams
        

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
        out["primitive_params_"] = self.primitive_params_

        return out