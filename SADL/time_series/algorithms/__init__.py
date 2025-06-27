#from . import tods
from . import tsfedl

__all__ = ['tsfedl']

def get_algorithms():
    return {'components': __all__}