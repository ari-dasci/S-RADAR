from . import images
from . import static_data
from . import time_series

__all__ = ['images', 'static_data', 'time_series']

def get_components():
    return {'components': __all__}

 