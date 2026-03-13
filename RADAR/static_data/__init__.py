from . import algorithms
from . import anomaly_dataset_utils
from . import preprocessing


__all__ = ['algorithms', 'anomaly_dataset_utils', 'preprocessing']


def get_categories():
    return {'categories': __all__}