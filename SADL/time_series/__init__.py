from . import algorithms
from . import preprocessing
from . import visualization

__all__ = ['algorithms', 'preprocessing', 'visualization']

def get_components():
    return {'components': __all__}