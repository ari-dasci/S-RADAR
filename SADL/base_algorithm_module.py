"""
    Base Module that performs the main operations needed to adapt every anomaly detection library

"""
from inspect import signature
from collections import defaultdict
from abc import ABC, abstractmethod
import inspect

class BaseAnomalyDetection(ABC):
    """
    Abstract Base class for every AD library to define the main behaviour of every library and specific algorithm

    Attributes
    ----------
    decision_scores : numpy array of shape (n_samples,)
        The outlier scores of the training data 
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    label_parser : function of shape (n_samples,) with the 
        specific methods or operations to apply to the score values.

    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        self.label_parser = kwargs.get('label_parser', None)

    @abstractmethod
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
        pass

    @abstractmethod
    def decision_function(self, X):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. 

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """
        pass

    @abstractmethod
    def get_params(self):
        """Get parameters for this estimator.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return ['label_parser']

    @abstractmethod
    def set_params(self, **params):
        """Set the parameters of this estimator.
        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information. 

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        
        valid_params = self.get_params()

        for key, value in params.items():
            if key == "label_parser":
                setattr(self, key, value)

        return self
