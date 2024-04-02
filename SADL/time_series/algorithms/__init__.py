from . import tods
from . import tsfedl

__all__ = ['tods', 'tsfedl']

def get_algorithms():
    return {'components': __all__}