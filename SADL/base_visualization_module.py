"""
    Base Visualization Module that performs the main operations needed to visualize every type of data

"""
from abc import ABC, abstractmethod

class BaseVisualization(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass
    
    @abstractmethod
    def show(self):
        pass

    # [...] other possible methods