from SADL.base_algorithm_module import BaseAnomalyDetection
from SADL.time_series.algorithms.modelsTransformersTS.vanillaTransformer.model import Transformer
from SADL.time_series.algorithms.modelsTransformersTS.informer.model import Informer
from SADL.time_series.algorithms.modelsTransformersTS.autoformer.model import Autoformer
from inspect import signature
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from SADL.metrics_module import print_metrics


transformers_algorithms = {
    "Transformer": Transformer, 
    "Informer": Informer,
    "Autoformer":Autoformer
}

class TransformersAnomalyDetection(BaseAnomalyDetection):
    def __init__(self, **kwargs):
        self.model = None
        self.algorithm_name = kwargs.get("algorithm_", "Transformer")
        self.algorithm_ = transformers_algorithms[self.algorithm_name]
        self.set_params(**kwargs)
        
        
        # Training parameters we get with set_params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = self.train_params.get("lr", 1e-3)
        self.train_epochs = self.train_params.get("train_epochs", 10)
        self.batch_size = self.train_params.get("batch_size", 32)
        self.label_parser = self.train_params.get("label_parser", None)
           
        self.optimizer = None
        self.criterion = None  

    
    def fit(self, X, y=None):
        """Train Transformer model. `X` is the input sequence. If `y` is provided, it is used as target"""
        self.model.to(self.device)
        self.model.train()

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)

        if y is not None:
            if isinstance(y, np.ndarray):
                y = torch.tensor(y, dtype=torch.float32)
            elif isinstance(y, pd.Series):
                y = torch.tensor(y.values, dtype=torch.float32)
            dataset = TensorDataset(X, y)
        else:
            dataset = TensorDataset(X)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        for epoch in range(self.train_epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                if y is not None:
                    inputs, targets = batch
                else:
                    inputs = batch[0]
                    targets = inputs  # reconstruction

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Build decoder_input by shifting targets to the right
                decoder_inputs = torch.zeros_like(targets)
                decoder_inputs[:, 1:] = targets[:, :-1]
                decoder_inputs[:, 0] = 0  # or use startup token if applicable
                
                self.optimizer.zero_grad()
                #forward for Transformer with decoder input
                outputs, *_ = self.model(inputs, decoder_inputs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.train_epochs}, Loss: {epoch_loss / len(train_loader):.6f}")

        return self

    def predict(self, X):
        """Predicts outputs for input data X"""
        self.model.eval()

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)

        test_loader = DataLoader(TensorDataset(X), batch_size=1, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in test_loader:
                
                inputs = batch[0]
                targets = inputs  # reconstruction
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # Build decoder_input by shifting targets to the right
                decoder_inputs = torch.zeros_like(targets)
                decoder_inputs[:, 1:] = targets[:, :-1]
                decoder_inputs[:, 0] = 0  # or use startup token if applicable
                
                outputs, *_ = self.model(inputs,decoder_inputs)
                preds.append(outputs.cpu())
        return torch.cat(preds, dim=0)

    def decision_function(self, X):
        """Calculates anomaly scores (e.g., reconstruction error)."""
        self.model.eval()

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)

        test_loader = DataLoader(TensorDataset(X), batch_size=1, shuffle=False)

        scores = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0]
                targets = inputs  # reconstruction
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Build decoder_input by shifting targets to the right
                decoder_inputs = torch.zeros_like(targets)
                decoder_inputs[:, 1:] = targets[:, :-1]
                decoder_inputs[:, 0] = 0  # or use startup token if applicable
                
                outputs, *_ = self.model(inputs,decoder_inputs)
                # Mean absolute error por muestra
                batch_scores = torch.mean(torch.abs(inputs - outputs), dim=(1, 2))
                scores.append(batch_scores.cpu())
        return torch.cat(scores, dim=0)


    def score_to_label_fn(self, scores):
        threshold = np.mean(scores) + 3 * np.std(scores)
        #np.percentile(scores, 95)
        return (scores > threshold).astype(int)
         
    
    def set_params(self, **params):
        """Set the parameters of this estimator.
        Returns
        -------
        self : object
        """
        super().set_params(**params)
        if not params:
            return self
        
        # Separate train parameters (do not belong to the model)
        self.train_params = {
            k: v for k, v in params.items() if k in ["train_epochs", "batch_size", "lr","label_parser"]
        }

        # Model parameters (all other)
        model_params = {
            k: v for k, v in params.items() if k not in self.train_params
        }

        print(f"Train Params: {self.train_params} \nModel Params: {model_params}")

        if self.algorithm_name not in transformers_algorithms:
            raise ValueError(f"The algorithm '{self.algorithm_name}' is not defined.")

        # Obtain valid model parameters
        valid_params = self.get_default_params(**model_params)
        # Assign algorithm name
        setattr(self.algorithm_, "algorithm_", valid_params["algorithm_"])
        
        for key in model_params.keys():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for model {self.algorithm_name}. "
                    f"Valid parameters are: {valid_params.keys()!r}."
                )

        # Identify mandatory positional parameters of the model
        positional_params = {}
        init_signature = signature(self.algorithm_.__init__)
        for param_name, param in init_signature.parameters.items():
            if param_name != "self":
                if param_name in model_params:
                    positional_params[param_name] = model_params[param_name]
        
        # Init Model
        try:
            self.model = self.algorithm_(**positional_params)
        except Exception as e:
            print("Error instantiating the model:", str(e))
            raise
        # Assign the remaining parameters to the model 
        for key, value in model_params.items():
            setattr(self.model, key, value)

        return self

    def get_default_params(self, **params):
        """Get DEFAULT parameters for this estimator, params is used to configure positional parameters in order to
        obtain default parameters of the object.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
          
        train_params = {
            k: v for k, v in params.items() if k in ["train_epochs", "batch_size", "lr","label_parser"]
        }
        model_params = {k: v for k, v in params.items() if k not in train_params}

        out = super().get_params()
               
        init_signature = signature(self.algorithm_.__init__)
        param_names = [
            p.name
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        ]
       
        # Identify the positional parameters required for the model.
        positional_params = {
            p.name: model_params[p.name]
            for p in init_signature.parameters.values()
            if p.name in model_params
        }
        
        # Use the existing model or instantiate a new one if necessary.
        model_instance = self.model or self.algorithm_(**positional_params)
        
        # Obtain model parameters
        out["algorithm_"] = self.algorithm_name
        
        for key in param_names:
            out[key] = getattr(model_instance, key, None)
        
        return out


    def get_params(self,**kwargs):
        """Get parameters for this estimator.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = super().get_params()
        out["algorithm_"] = self.algorithm_name

        if not self.model:
            return out  # If there is no model, return only the basics

        # Get parameters directly from the model

        if hasattr(self.model, "get_params"):
            out.update(self.model.get_params())
        else:
            print(
                f"Warning: The model '{self.algorithm_name}' does not have a 'get_params' method."
            )
            print("Inspecting model's attributes:")
            model_attributes = vars(self.model)  # This returns the attribute dictionary
            for attr, value in model_attributes.items():
                print(f"{attr}: {value}")
            out.update(model_attributes)

        return out
    
    
    
    def evaluate(self, X, y=None, threshold=None):
        """
        Evaluates the model on data X (and optionally y).
        If no threshold is provided, it is calculated as mean + 3 * std.
        """

        self.model.eval()

        if isinstance(X, (np.ndarray, pd.DataFrame)):
            X = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32)
        if y is not None and isinstance(y, (np.ndarray, pd.DataFrame)):
            y = torch.tensor(y.values if isinstance(y, pd.DataFrame) else y, dtype=torch.int64)
            
        dataset = TensorDataset(X) 
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        y_true = []
        y_scores = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)

                decoder_inputs = torch.zeros_like(inputs)
                decoder_inputs[:, 1:] = inputs[:, :-1]
                decoder_inputs[:, 0] = 0  
                
                
                outputs, *_ = self.model(inputs,decoder_inputs)

            # MSE reconstruction error
                score = torch.mean((outputs - inputs) ** 2, dim=-1)
                y_scores.append(score.detach().cpu())
                
        
        y_scores = torch.cat(y_scores).numpy().flatten()
        self.labels_preds = self.label_parser(y_scores) if hasattr(self, "label_parser") and self.label_parser else self.score_to_label_fn(y_scores)         
             
        results = {
            "scores": y_scores,
            "labels_preds": self.labels_preds,
        }
       
        if y is not None:
            y_true = y.flatten()
            print_metrics(["Accuracy", "F1", "Recall", "Precision"], y_true, self.labels_preds)
            results["labels_true"] = y_true  
            
        return results