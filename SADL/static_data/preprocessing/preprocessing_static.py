from SADL.base_preprocessing_module import BasePreprocessing
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
    OneHotEncoder,
)
import pandas as pd

preprocessing_static_algorithms = {
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "Normalizer",
    "OneHotEncoder"
}

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
        X_encoded = self.encoder.transform(X_selected).astype(int)
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