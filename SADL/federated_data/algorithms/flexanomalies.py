from SADL.base_algorithm_module import BaseAnomalyDetection
from flexanomalies.utils import AutoEncoder
from flexanomalies.utils import IsolationForest
from flexanomalies.utils import PCA_Anomaly
from flexanomalies.utils import ClusterAnomaly
from flexanomalies.utils import DeepCNN_LSTM
from flexanomalies.utils.load_data import federate_data
from flexanomalies.pool.aggregators_favg import aggregate_ae
from flexanomalies.pool.aggregators_cl import aggregate_cl
from flexanomalies.pool.aggregators_pca import aggregate_pca
from flexanomalies.pool.primitives_deepmodel import (
    build_server_model_ae,
    copy_model_to_clients_ae,
    train_ae,
    set_aggregated_weights_ae,
    weights_collector_ae,
    evaluate_global_model,
)
from flexanomalies.pool.primitives_cluster import (
    build_server_model_cl,
    copy_model_to_clients_cl,
    train_cl,
    set_aggregated_weights_cl,
    get_clients_weights_cl,
)
from flexanomalies.pool.primitives_iforest import (
    build_server_model_if,
    copy_model_to_clients_if,
    train_if,
    aggregate_if,
    set_aggregated_weights_if,
    get_clients_weights_if,
)
from flexanomalies.pool.primitives_pca import (
    build_server_model_pca,
    copy_model_to_clients_pca,
    train_pca,
    set_aggregated_weights_pca,
    get_clients_weights_pca,
)
from flexanomalies.utils.save_results import save_experiments_results
from flex.pool import FlexPool
from flexanomalies.utils.metrics import *
from inspect import signature

flexanomalies_algorithms = {
    "autoencoder": AutoEncoder,
    "isolationForest": IsolationForest,
    "pcaAnomaly": PCA_Anomaly,
    "clusterAnomaly": ClusterAnomaly,
    "deepCNN_LSTM": DeepCNN_LSTM,
}


class FlexAnomalyDetection(BaseAnomalyDetection):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.algorithm_name = kwargs.get("algorithm_", "autoencoder")
        self.algorithm_ = flexanomalies_algorithms[self.algorithm_name]
        self.model = None
        self.pool = None
        self.flex_data = None
        self.federated_functions = {
            "autoencoder": {
                "build_model": build_server_model_ae,
                "copy": copy_model_to_clients_ae,
                "train": train_ae,
                "collect": weights_collector_ae,
                "aggregate": aggregate_ae,
                "set_weights": set_aggregated_weights_ae,
            },
            "isolationForest": {
                "build_model": build_server_model_if,
                "copy": copy_model_to_clients_if,
                "train": train_if,
                "collect": get_clients_weights_if,
                "aggregate": aggregate_if,
                "set_weights": set_aggregated_weights_if,
            },
            "pcaAnomaly": {
                "build_model": build_server_model_pca,
                "copy": copy_model_to_clients_pca,
                "train": train_pca,
                "collect": get_clients_weights_pca,
                "aggregate": aggregate_pca,
                "set_weights": set_aggregated_weights_pca,
            },
            "clusterAnomaly": {
                "build_model": build_server_model_cl,
                "copy": copy_model_to_clients_cl,
                "train": train_cl,
                "collect": get_clients_weights_cl,
                "aggregate": aggregate_cl,
                "set_weights": set_aggregated_weights_cl,
            },
            "deepCNN_LSTM": {
                "build_model": build_server_model_ae,
                "copy": copy_model_to_clients_ae,
                "train": train_ae,
                "collect": weights_collector_ae,
                "aggregate": aggregate_ae,
                "set_weights": set_aggregated_weights_ae,
            },
        }
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
        if name in flexanomalies_algorithms:
            print(f" The algorithm {name} is already registered and will be overwritten.")
        flexanomalies_algorithms[name] = model_class
            
    def register_federated_functions(self, name, functions):
        """Records the federated functions of a new algorithm.
             - federated_functions (dict): A dictionary defining federated learning functions.
         Required keys:
            - "build_model": Function to initialize the server model.
            - "copy": Function to distribute the model to clients.
            - "train": Function to train the model on client data.
            - "collect": Function to gather updates from clients.
            - "aggregate": Function to combine client updates.
            - "set_weights": Function to update the global model.          
        """
        self.federated_functions[name] = functions
    
    def fit(self, X, y):
        """Fit detector.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape


        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.algorithm_name not in self.federated_functions:
            raise ValueError(
                f"No federated functions defined for {self.algorithm_name},model not defined"
            )

        # Define the pool for the federated system
        self._initialize_pool(X, y)

        federated_ops = self.federated_functions[self.algorithm_name]

        for i in range(self.n_rounds):
            print(f"\nRunning round: {i}\n")
            self.pool.servers.map(federated_ops["copy"], self.pool.clients)
            self.pool.clients.map(federated_ops["train"])
            self.pool.aggregators.map(federated_ops["collect"], self.pool.clients)
            if self.algorithm_name in ["clusterAnomaly", "ClusterAnomaly"]:
                self.pool.aggregators.map(federated_ops["aggregate"], model=self.model)
            else:
                self.pool.aggregators.map(federated_ops["aggregate"])

            self.pool.aggregators.map(federated_ops["set_weights"], self.pool.servers)

        # Save model
        self.model = self.pool.servers._models[f"{self.algorithm_name}_server"]["model"]
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
        if self.model is None:
            raise ValueError("The model must be trained before using decision_function")
        try:
            return self.model.decision_function(X)
        except Exception as e:
            print(f"{self.algorithm_name}, decision_function():", str(e))

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
                print(f"{self.algorithm_name}, predict():", str(e))

    def evaluate(self, X_test, y_test, label_test=None):
        if self.model is None:
            raise ValueError("The model must be trained before evaluate.")
        
        if label_test is not None and label_test.size > 0:
            return evaluate_global_model(self.model, X_test, y_test, label_test)

        return self.model.evaluate(X_test, y_test)

    def set_params(self, **params):
        """Set the parameters of this estimator.
        Returns
        -------
        self : object
        """
        super().set_params(**params)
        # Simple optimization to gain speed (inspect is slow)
        if not params:
            return self

        # Separate federated parameters (do not belong to the model)
        self.federated_params = {
            k: v for k, v in params.items() if k in ["n_clients", "n_rounds"]
        }

        # Model parameters (all other)

        model_params = {
            k: v for k, v in params.items() if k not in self.federated_params
        }
        print(
            f"Federated Params:{self.federated_params} \n Model Params:{model_params}"
        )

        if self.algorithm_name not in flexanomalies_algorithms:
            raise ValueError(f"The algorithm '{self.algorithm_name}' is not defined.")

        # Obtain valid model parameters
        valid_params = self.get_default_params(**model_params)
        
        # Assign algorithm name
        setattr(self.algorithm_, "algorithm_", valid_params["algorithm_"])

        # Verification of invalid parameters for the model
        for key, value in model_params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for model {self.algorithm_name}. "
                    f"Valid parameters are: {valid_params!r}."
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
            print("Error al instanciar el modelo:", str(e))
            raise

        # Assign the remaining parameters to the model
        for key, value in model_params.items():
            setattr(self.model, key, value)

        return self

    def _initialize_pool(self, X_train=None, y_train=None):

        self.n_clients = self.federated_params["n_clients"]
        self.n_rounds = self.federated_params["n_rounds"]
        
        
        #If flex data is an attribute, then we use the federated data set given as attribute
        if hasattr(self, "flex_data") and self.flex_data is not None:
            print("Using external federated dataset.")
            flex_dataset = self.flex_data  
        else:
            X, y = X_train, y_train
            flex_dataset = federate_data(self.n_clients, X, y)
            
        federated_ops = self.federated_functions.get(self.algorithm_name)
        if not federated_ops:
            raise ValueError(
                f"No federated functions defined for {self.algorithm_name}"
            )

        # Define pool for clients and server
        self.pool = FlexPool.client_server_pool(
            fed_dataset=flex_dataset,
            server_id=f"{self.algorithm_name}_server",
            init_func=federated_ops["build_model"],
            model=self.model,
        )

    def get_default_params(self, **params):
        """Get DEFAULT parameters for this estimator, params is used to configure positional parameters in order to
        obtain default parameters of the object.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        # Separating federated parameters from model parameters
        federated_params = {
            k: v for k, v in params.items() if k in ["n_clients", "n_rounds"]
        }
        model_params = {k: v for k, v in params.items() if k not in federated_params}

        
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

    def get_params(self):
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
