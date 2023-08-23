import logging
import pandas as pd
from pydantic import BaseModel
import joblib
import numpy as np
import pytest

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

def test_data_existence():
    # Load training data
    logger.debug('Loading training data')
    df_train = pd.read_csv(r'Project\finalproject_itesm_mlops\data\fraud_oracle.csv')

    # Verify training data existence
    assert not df_train.empty, "No training data"
    logger.info('Training data exists')

class InputData(BaseModel):
    feature1: float
    feature2: float
    # Add more features as needed

class OutputData(BaseModel):
    prediction: int

model = joblib.load(r'Project\finalproject_itesm_mlops\models\dtmodel.pkl')

app = None

@pytest.fixture(scope="module")
def client():
    global app
    from fastapi.testclient import TestClient
    from main import app
    yield TestClient(app)

def test_predict(client):
    input_data = {"feature1": 1.0, "feature2": 2.0}
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": 0}  # Update expected prediction value based on your model

if __name__ == '__main__':
    pytest.main(["-v"])