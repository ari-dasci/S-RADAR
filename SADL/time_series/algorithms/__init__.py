#from . import tods
from . import tsfedl
from . import transformers

__all__ = ['tods', 'tsfedl','transformers']

def get_algorithms():
    return {'components': __all__}