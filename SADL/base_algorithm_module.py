"""
    Base Module that performs the main operations needed to adapt every anomaly detection library

"""
from abc import ABC, abstractmethod

class BaseAnomalyDetection(ABC):
    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def decision_function(self, data):
        pass

    # [...] other possible methods