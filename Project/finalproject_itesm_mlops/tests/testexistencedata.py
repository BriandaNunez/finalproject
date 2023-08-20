import logging
import pytest
import pandas as pd

# Set up logging configuration
log_file = r'C:\Users\brianda.nunez\Documents\GitHub\finalproject\Project\finalproject_itesm_mlops\tests\testexistencedata.log'
logging.basicConfig(filename=log_file, level=logging.INFO)

def test_data_existence():
    # Load training data
    df_train = pd.read_csv(r'Project\finalproject_itesm_mlops\data\fraud_oracle.csv')

    # Verify training data existence
    if df_train.empty:
        logging.error("No training data")
        assert not df_train.empty, "No training data"
    else:
        logging.info("Training data loaded successfully")
