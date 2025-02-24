from SADL.base_preprocessing_module import BasePreprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer,OneHotEncoder
import numpy as np

class StandardScalerPreprocessing(BasePreprocessing):
    def __init__(self,**kwargs):
        self.scaler = StandardScaler(**kwargs)
    
    def fit(self, X):
        self.scaler.fit(X)
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)


class MinMaxScalerPreprocessing(BasePreprocessing):
    def __init__(self,**kwargs):
        self.scaler = MinMaxScaler(**kwargs)
    
    def fit(self, X):
        self.scaler.fit(X)
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

class RobustScalerPreprocessing(BasePreprocessing):
    def __init__(self,**kwargs):
        self.scaler = RobustScaler(**kwargs)
    
    def fit(self, X):
        self.scaler.fit(X)
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)


class NormalizerPreprocessing(BasePreprocessing):
    def __init__(self,**kwargs):
        self.scaler = Normalizer(**kwargs)
    
    def fit(self, X):
        self.scaler.fit(X)
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def inverse_transform(self, X):
        raise NotImplementedError("Normalizer does not support inverse transform")


class OneHotEncoderPreprocessing(BasePreprocessing):
    def __init__(self,**kwargs):
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore',**kwargs)
    
    def fit(self, X):
        self.encoder.fit(X)
    
    def transform(self, X):
        return self.encoder.transform(X)
    
    def inverse_transform(self, X):
        return self.encoder.inverse_transform(X) 
    
    
class RollingMeanProcessing(BasePreprocessing):
    def __init__(self, window=3, **kwargs):
        super().__init__(window=window, **kwargs)
        self.window = window
    
    def fit(self, X):
        pass  # No requiere ajuste
    
    def transform(self, X):
        """
        Apply rolling mean transformation.
        
        Input:
        - X: pandas.DataFrame or pandas.Series, where rows are time steps 
             and columns are the different time series.
        
        Output:
        - pandas.DataFrame or pandas.Series with rolling mean applied over 
          the specified window.
        """
        return X.rolling(window=self.window, min_periods=1).mean()
    
    def inverse_transform(self, X):
        raise NotImplementedError("Rolling mean is not invertible")

class InterpolationProcessing(BasePreprocessing):
    def __init__(self, method='linear', **kwargs):
        super().__init__(method=method, **kwargs)
        self.method = method
    
    def fit(self, X):
        pass  # No requiere ajuste
    
    def transform(self, X):
        """
        Apply interpolation to fill missing values.
        
        Input:
        - X: pandas.DataFrame or pandas.Series, where rows are time steps 
             and columns may contain missing values that need to be interpolated.
        
        Output:
        - pandas.DataFrame or pandas.Series with interpolated values, using 
          the specified interpolation method (e.g., 'linear', 'polynomial').
        """
        return X.interpolate(method=self.method)
    
    def inverse_transform(self, X):
        raise NotImplementedError("Interpolation is not invertible")


class FilterProcessing(BasePreprocessing):
    def __init__(self, filter_func=np.mean, kernel_size=3, **kwargs):
        super().__init__(filter_func=filter_func, kernel_size=kernel_size, **kwargs)
        self.filter_func = filter_func
        self.kernel_size = kernel_size
    
    def fit(self, X):
        pass  # No requiere ajuste
    
    def transform(self, X):
        """
        Apply filtering using a moving average kernel.
        
        Input:
        - X: numpy.ndarray or pandas.Series, representing a one-dimensional 
             time series. If a DataFrame is provided, each column will be processed independently.
        
        Output:
        - numpy.ndarray or pandas.Series with the filtered series, using 
          a moving average filter with the specified kernel size.
        """
        return np.convolve(X, np.ones(self.kernel_size)/self.kernel_size, mode='same')
    
    def inverse_transform(self, X):
        raise NotImplementedError("Filtering is not invertible")
    