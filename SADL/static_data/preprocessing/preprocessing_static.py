from SADL.base_preprocessing_module import BasePreprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer,OneHotEncoder


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