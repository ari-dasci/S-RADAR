from SADL.base_preprocessing_module import BasePreprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer,OneHotEncoder


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

class OneHotEncoderPreprocessing(BasePreprocessing):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', **kwargs)
    
    def fit(self, X):
        self.encoder.fit(X)
        return self
    
    def transform(self, X):
        return self.encoder.transform(X)
    
    def fit_transform(self, X):
        return self.encoder.fit_transform(X)
    
    def inverse_transform(self, X):
        return self.encoder.inverse_transform(X)