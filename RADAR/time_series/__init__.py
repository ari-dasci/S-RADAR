from . import algorithms
from . import preprocessing

__all__ = ['algorithms', 'preprocessing']

def get_categories():
    return {'components': __all__}