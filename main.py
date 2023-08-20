import logging
import unittest
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Configure the logger
logging.basicConfig(filename='C:\\Users\\brianda.nunez\\Documents\\GitHub\\finalproject\\main.log', encoding='utf-8', level=logging.DEBUG)

# Create a logger object
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Add a console handler and set its level to DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the console handler
ch.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(ch)

class TestDataExistence(unittest.TestCase):
    def test_data_existence(self):
        # Load training data
        logger.debug('Loading training data')
        df_train = pd.read_csv(r'Project\finalproject_itesm_mlops\data\fraud_oracle.csv')

        # Verify training data existence
        self.assertFalse(df_train.empty, "No training data")
        logger.info('Training data exists')

class InputData(BaseModel):
    feature1: float
    feature2: float
    # Add more features as needed

class OutputData(BaseModel):
    prediction: int

model = joblib.load(r'Project\finalproject_itesm_mlops\models\dtmodel.pkl')

# Add FastAPI to the project

app = FastAPI()

@app.post("/train")
def train_model():
    # Implement the logic to train a new model
    # Return a response indicating the success or failure of the training process
    logger.info('Training model')
    pass

@app.post("/predict", response_model=OutputData)
def predict(input_data: InputData):
    # Convert the input data to a numpy array
    input_array = np.array([[input_data.feature1, input_data.feature2]])

    # Make predictions using the loaded model
    prediction = model.predict(input_array)

    # Create an instance of the output data model and return it
    output_data = OutputData(prediction=int(prediction[0]))
    return output_data

if __name__ == "__main__":
    unittest.main()