from . import algorithms
from . import preprocessing

__all__ = ['algorithms', 'preprocessing']

def get_components():
    return {'components': __all__}