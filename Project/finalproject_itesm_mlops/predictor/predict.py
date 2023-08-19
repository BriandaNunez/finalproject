import logging
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

class ModelPredictor:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.setup_logging()
        self.load_model()

    def setup_logging(self):
        log_file_path = r'C:\Users\brianda.nunez\Documents\GitHub\finalproject\Project\finalproject_itesm_mlops\predictor\predict.log'
        logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        try:
            self.logger.debug(f"Cargando el modelo desde: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            self.logger.debug("Modelo cargado correctamente")
        except Exception as e:
            self.logger.critical(f"Error al cargar el modelo: {str(e)}")

    def predict(self, new_data):
        try:
            if new_data is None:
                warnings.warn("No se ha cargado `new_data`. La predicción podría dar resultados incorrectos.", Warning)
            else:
                # Applying StandardScaler
                SC = StandardScaler()
                new_data = pd.DataFrame(SC.fit_transform(new_data.values),
                                        index=new_data.index, columns=new_data.columns)
                return self.model.predict(new_data)
        except Exception as e:
            self.logger.error(f"Error al realizar la predicción: {str(e)}")

if __name__ == '__main__':
    model_path = r'C:\Users\brianda.nunez\Documents\GitHub\finalproject\Project\finalproject_itesm_mlops\models\dtmodel.pkl'  # Reemplaza con la ruta correcta al archivo .pkl
    predictor = ModelPredictor(model_path)
    new_data = None  # Replace `None` with the actual data for prediction
    prediction = predictor.predict(new_data)

    if prediction is None:
        print("No prediction. Please make sure `new_data` is loaded.")
    else:
        # Do something with the prediction
        pass

