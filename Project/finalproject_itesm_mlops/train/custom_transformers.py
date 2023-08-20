import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Configurar el logging
log_file_path = r'C:\Users\brianda.nunez\Documents\GitHub\finalproject\Project\finalproject_itesm_mlops\train\custom_transformers.log'
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        logger.debug("Fitting DropColumnsTransformer")
        return self

    def transform(self, X):
        logger.debug("Transforming with DropColumnsTransformer")
        X = X.drop(columns=self.columns)
        return X

class FillNaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        logger.debug("Fitting FillNaTransformer")
        return self

    def transform(self, X):
        logger.debug("Transforming with FillNaTransformer")
        for column in self.columns:
            X[column].fillna(X[column].mean(), inplace=True)
        return X

class OneHotEncodingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        logger.debug("Fitting OneHotEncodingTransformer")
        return self

    def transform(self, X):
        logger.debug("Transforming with OneHotEncodingTransformer")
        return pd.get_dummies(X[self.columns])

class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        logger.debug("Fitting StandardScalerTransformer")
        return self

    def transform(self, X):
        logger.debug("Transforming with StandardScalerTransformer")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X[self.columns])
        for i in range(len(self.columns)):
            X[self.columns[i]] = scaled_data[:, i]
        return X
