from SADL.base_preprocessing_module import BasePreprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer,OneHotEncoder
import numpy as np
import pandas as pd


class StandardScalerPreprocessing(BasePreprocessing):
    def __init__(self, **kwargs):
        super().__init__()
        self.scaler = StandardScaler(**kwargs)
    
    def fit(self, X):
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)


class MinMaxScalerPreprocessing(BasePreprocessing):
    def __init__(self, **kwargs):
        super().__init__()
        self.scaler = MinMaxScaler(**kwargs)
    
    def fit(self, X):
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)


class RobustScalerPreprocessing(BasePreprocessing):
    def __init__(self, **kwargs):
        super().__init__()
        self.scaler = RobustScaler(**kwargs)
    
    def fit(self, X):
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)


class NormalizerPreprocessing(BasePreprocessing):
    def __init__(self, **kwargs):
        super().__init__()
        self.scaler = Normalizer(**kwargs)
    
    def fit(self, X):
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        return self.scaler.fit_transform(X)
    
    def inverse_transform(self, X):
        raise NotImplementedError("Normalizer does not support inverse transform")
    
class RollingMeanPreprocessing(BasePreprocessing):
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

class InterpolationPreprocessing(BasePreprocessing):
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


class FilterPreprocessing(BasePreprocessing):
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

class OneHotEncoderPreprocessing(BasePreprocessing):
    def __init__(self, columns=None, **kwargs):
        """
        Parameters
        ----------
        columns : list, optional
            List of column names to apply one-hot encoding.
        """
        super().__init__()
        self.columns = columns
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", **kwargs)
        self.feature_names = None

    def fit(self, X):
        X_selected = X[self.columns] if self.columns else X
        self.encoder.fit(X_selected)
        self.feature_names = self.encoder.get_feature_names_out(self.columns)
        return self

    def transform(self, X):
        X_selected = X[self.columns] if self.columns else X
        X_encoded = self.encoder.transform(X_selected)
        X_encoded_df = pd.DataFrame(X_encoded, columns=self.feature_names, index=X.index)
        X_remaining = X.drop(columns=self.columns, errors='ignore')
        return X_remaining.join(X_encoded_df)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X_encoded = X[self.feature_names]
        X_decoded = self.encoder.inverse_transform(X_encoded)
        X_decoded_df = pd.DataFrame(X_decoded, columns=self.columns, index=X.index)
        X_remaining = X.drop(columns=self.feature_names, errors='ignore')
        return X_remaining.join(X_decoded_df)    
    

preprocessing_ts_algorithms = {
    "StandardScaler": StandardScalerPreprocessing,
    "MinMaxScaler": MinMaxScalerPreprocessing,
    "RobustScaler": RobustScalerPreprocessing,
    "Normalizer": NormalizerPreprocessing,
    "RollingMean": RollingMeanPreprocessing,
    "Interpolation": InterpolationPreprocessing,
    "Filter": FilterPreprocessing,
    "OneHotEncoder": OneHotEncoderPreprocessing
}