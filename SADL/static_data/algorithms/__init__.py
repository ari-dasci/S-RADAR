from . import pyod
from . import sklearn

__all__ = ['pyod', 'sklearn']

def get_algorithms():
    return {'components': __all__}