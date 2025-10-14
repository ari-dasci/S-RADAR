from . import algorithms
from . import preprocessing


__all__ = ['algorithms', 'preprocessing']


def get_categories():
    return {'categories': __all__}