from . import tods
from . import tsfedl

__all__ = ['preprocessing_ts']

def get_algorithms():
    return {'components': __all__}