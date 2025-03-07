from . import images
from . import static_data
from . import time_series
from . import federated_data

__all__ = ['images', 'static_data', 'time_series','federated_data']

def get_components():
    return {'components': __all__}

 