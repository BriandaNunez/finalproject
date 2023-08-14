# predictor.py
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ModelPredictor:

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(r'Project\finalproject_itesm_mlops\models\dtmodel.pkl')

    def predict(self, new_data):
        # Applying StandardScaler
        SC = StandardScaler()
        new_data = pd.DataFrame(SC.fit_transform(new_data.values),
                                index=new_data.index, columns=new_data.columns)
        return self.model.predict(new_data)
