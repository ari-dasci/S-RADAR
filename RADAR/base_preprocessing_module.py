"""
    Base Preprocessing Module that performs the main operations needed to preprocess every type of data

"""
from abc import ABC, abstractmethod

class BasePreprocessing(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass
    
    @abstractmethod
    def inverse_transform(self, X):
        pass 
    # [...] other possible methods