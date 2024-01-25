# Types of parameters for every model in pyod library and its security requirements in order to be correct
PYOD_PARAMETERS = {
    'ABOD': {'value':'contamination', 'min': 0, 'max': 1, 'type': 'float'},
}