"""
    Base Visualization Module that performs the main operations needed to visualize every type of data

"""
from abc import ABC, abstractmethod

class BaseVisualization(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    # [...] other possible methods