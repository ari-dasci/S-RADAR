from . import tsfedl
from . import transformers

__all__ = ['tsfedl','transformers']

def get_algorithms():
    return {'components': __all__}