from SADL.base_algorithm_module import BaseAnomalyDetection
import pytorch_lightning as pl
import numpy as np
import traceback
from torch.nn import functional as F
import torch
import torch.nn as nn
from inspect import signature
from torch.utils.data import DataLoader, TensorDataset
from TSFEDL.models_pytorch import OhShuLih
from TSFEDL.models_pytorch import YiboGao
from TSFEDL.models_pytorch import LihOhShu
from TSFEDL.models_pytorch import YaoQihang
from TSFEDL.models_pytorch import HtetMyetLynn
from TSFEDL.models_pytorch import YildirimOzal
from TSFEDL.models_pytorch import CaiWenjuan
from TSFEDL.models_pytorch import ZhangJin
from TSFEDL.models_pytorch import KongZhengmin
from TSFEDL.models_pytorch import WeiXiaoyan
from TSFEDL.models_pytorch import GaoJunLi
from TSFEDL.models_pytorch import KhanZulfiqar
from TSFEDL.models_pytorch import ZhengZhenyu
from TSFEDL.models_pytorch import WangKejun
from TSFEDL.models_pytorch import ChenChen
from TSFEDL.models_pytorch import KimTaeYoung
from TSFEDL.models_pytorch import GenMinxing
from TSFEDL.models_pytorch import FuJiangmeng
from TSFEDL.models_pytorch import ShiHaotian
from TSFEDL.models_pytorch import HuangMeiLing
from TSFEDL.models_pytorch import HongTan
from TSFEDL.models_pytorch import SharPar
from TSFEDL.models_pytorch import DaiXiLi
from TSFEDL.models_pytorch import TSFEDL_BaseModule
from SADL.metrics_module import print_metrics
import pandas as pd

tsfedl_algorithms = {
    "ohshulih" : OhShuLih,
    "yibogao": YiboGao,
    "liohshu": LihOhShu,
    "yaoqihang" : YaoQihang,
    "htetmyetlynn" : HtetMyetLynn,
    "yildirimozal" : YildirimOzal,
    "caiwenjuan" : CaiWenjuan,
    "zhangjin" : ZhangJin,
    "kongzhengmin": KongZhengmin,
    "weixiaoyan" : WeiXiaoyan,
    "gaojunli": GaoJunLi,
    "khanzulfiqar" : KhanZulfiqar,
    "zhengzhenyu": ZhengZhenyu,
    "wangkejun" : WangKejun,
    "chenchen": ChenChen,
    "kimtaeyoung" : KimTaeYoung,
    "genminxing": GenMinxing,
    "fujiangmeng" : FuJiangmeng,
    "shihaotian" : ShiHaotian, 
    "huangmeiling": HuangMeiLing,
    "hongtan": HongTan,
    "sharpar" : SharPar,
    "daixili": DaiXiLi
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
        
        pytorch_params_: dict object of pl params for the Trainer object. To see the particular attributes: https://lightning.ai/docs/pytorch/stable/common/trainer.html
    """
    def __init__(self,batch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.algorithm_ = tsfedl_algorithms[kwargs.get('algorithm_', 'ohshulih')] #Default to OhShuLih 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None   # base model
        self.batch_size = batch_size
        self.pytorch_params_ = {}
        self.set_params(**kwargs)
                
    def fit(self, X, y=None):
        """
        Fit detector. y is ignored in unsupervised methods.

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
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        if y is not None and isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long)

        # Crear dataloader
        if y is not None:
            dataset = TensorDataset(X, y)
        else:
            dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        try:
            # Revisar si hay parámetros de trainer
            trainer_params = getattr(self, "pytorch_params_", {})
            trainer = pl.Trainer(**trainer_params) if trainer_params else pl.Trainer()
            trainer.fit(self.model, dataloader)
        except Exception as e:
            print("TSFEDLerror fit(): ", str(e))
            print("For further reference please see: https://s-tsfe-dl.readthedocs.io/en/latest/index.html")
            raise

        return self
     
    
    def decision_function(self, X):
        """
        Compute the reconstruction error of X using the trained model.
        This can be interpreted as a performance score of the prediction.

        Parameters
        ----------
        X : numpy array of shape (n_samples, seq_len, n_features)
            Input data.

        Returns
        -------
        scores : numpy array of shape (n_samples,)
            Reconstruction error per sample.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            X_pred = self.model(X)
            if isinstance(X_pred, torch.Tensor):
                X_pred = X_pred.cpu()

            # Error de reconstrucción (MSE por muestra)
            scores = torch.mean((X - X_pred) ** 2, dim=(1, 2)).numpy()

        return scores

    def predict(self, X,threshold=None):
        """
        Predict anomaly labels (0 = normal, 1 = anomaly).

        Parameters
        ----------
        X : array-like
            Input sequences.
        threshold : float, optional
            Threshold for classification. If None, uses mean + 3*std.

        Returns
        -------
        y_pred : numpy array of shape (n_samples,)
            Binary anomaly labels.
        """
        if "label_parser" in self.get_params().keys() and self.label_parser != None:
            return self.label_parser(X)
        else:
            scores = self.decision_function(X)

            if threshold is None:
                # Umbral simple: media + 3*desviación estándar
                threshold = np.mean(scores) + 3 * np.std(scores)

            y_pred = (scores > threshold).astype(int)
            return y_pred
        
        
    # def predict(self, X):
    #     """
    #     Generate raw predictions from the trained model.

    #     Parameters
    #     ----------
    #     X : numpy array or pandas DataFrame
    #         Input sequences.

    #     Returns
    #     -------
    #     y_pred : numpy array
    #         Model predictions with squeezed dimensions.
    #     """
    #     if isinstance(X, (np.ndarray, pd.DataFrame)):
    #         X = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32)

    #     self.model.eval()
    #     with torch.no_grad():
    #         y_pred = self.model(X.to(self.device))
    #         if isinstance(y_pred, torch.Tensor):
    #             y_pred = y_pred.cpu().numpy()

    #     # Remove singleton dimensions: (9618, 1, 4) -> (9618, 4)
    #     y_pred = np.squeeze(y_pred)

    #     return y_pred
    
    def evaluate(self, X, y=None, threshold=None):
        """
        Evaluate the model using decision_function (scores) and predict (outputs).

        Parameters
        ----------
        X : numpy array or pandas DataFrame
            Input sequences.
        y : numpy array or pandas DataFrame, optional
            Ground-truth values (for metrics).
        threshold : float, optional
            Threshold to convert scores into binary labels. 
            If None, it is computed as mean + 3 * std.

        Returns
        -------
        results : dict
            Dictionary containing:
                - "scores": anomaly/forecasting scores.
                - "labels_preds": predicted labels (if thresholding is applied).
                - "preds": model predictions.
                - "labels_true": ground-truth labels (if provided).
        """
        # Get raw scores from decision_function
        y_scores = self.decision_function(X)

        # Get predictions from predict
        preds = self.predict(X)

        # If thresholding applies (anomaly detection mode)
        if threshold is None:
            threshold = np.mean(y_scores) + 3 * np.std(y_scores)

        labels_preds = (
            self.label_parser(y_scores) if hasattr(self, "label_parser") and self.label_parser 
            else (y_scores > threshold).astype(int)
        )

        results = {
            "scores": y_scores,
            "labels_preds": labels_preds,
            "preds": preds,
        }

        if y is not None:
            y_true = y.flatten() if isinstance(y, np.ndarray) else y.values.flatten()
            results["labels_true"] = y_true

            # Example: print metrics if labels are available
            print_metrics(["Accuracy", "F1", "Recall", "Precision"], y_true, labels_preds)

        return results       
             
                   
    def set_params(self, **params): #Este setea sus propios parametros
        """
        Set the parameters of this estimator.
        Returns
        -------
        self : object
        """
        
        # --- Special for batch_size ---
        if "batch_size" in params:
            self.batch_size = params["batch_size"]
            del params["batch_size"]  # lo quitamos de params

        
        super().set_params(**params) #Llama al base para setear sus parametros en caso de que los hubiera

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        
        self.algorithm_ = tsfedl_algorithms[params.get('algorithm_', 'ohshulih')]# Default to OhShuLih 
        
        valid_params = self.get_default_params(**params)
        print(valid_params)
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
            #print(param_name)
            #print(params.keys())
            if param_name != 'self' and param.default == param.empty:  # Check for positional parameters
                if param_name in params.keys():
                    positional_params[param_name] = params[param_name]

        
        
        if not model_error:
            try:
                if "top_module" in params.keys():
                    positional_params["top_module"] = params["top_module"]
                    #self.in_features_top_module = params["top_module"].in_features  #NEWWWWWWWWW
                self.model = self.algorithm_(**positional_params)
            except Exception as e:
                print("TSFEDLerror:", str(e))
                print("For further reference please see: https://s-tsfe-dl.readthedocs.io/en/latest/index.html")
                raise

            for key, value in params.items():
                if key != "top_module":
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# class Model(TSFEDL_BaseModule):
#     def __init__(self, in_features_top_module, tsfedl_model, output_dim=1):
#         super().__init__(tsfedl_model.in_features)
        
#         # Modelo base (equivalente a include_top=False en Keras)
#         self.base = tsfedl_model  
#         # Bloque de capas densas según el paper
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(in_features_top_module, 20)   # ajusta input_dim según la salida de base
#         self.fc2 = nn.Linear(20, 10)
#         self.fc3 = nn.Linear(10, output_dim)  # salida para predicción
#     def forward(self, x):
#         x = self.base(x)
#         x = self.flatten(x)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)  # salida cruda, sin softmax ni sigmoid
#         return x