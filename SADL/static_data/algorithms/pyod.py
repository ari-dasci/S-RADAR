from ...base_algorithm_module import BaseAnomalyDetection
from pyod import pyod_detection_algorithm_example

class PyodAnomalyDetection(BaseAnomalyDetection):

    def predict(self, data):
        #Specific implementation
        pass

    
    def decision_function(self, data):
        #Specific implementation
        pass