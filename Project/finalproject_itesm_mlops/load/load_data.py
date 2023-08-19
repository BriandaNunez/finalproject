import pandas as pd
import logging
import os

# Get the absolute path of the log folder
log_folder = os.path.abspath(r'C:\Users\brianda.nunez\Documents\GitHub\finalproject\Project\finalproject_itesm_mlops\load')

# Create the log folder if it doesn't exist
os.makedirs(log_folder, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Create a file handler and set the level to DEBUG
file_handler = logging.FileHandler(os.path.join(log_folder, 'load_data.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Create a stream handler and set the level to INFO
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class DataLoader:
    """
    A class used to load data from a CSV file.

    ...

    Attributes
    ----------
    file_path : str
        Path to the CSV file.

    Methods
    -------
    load_data() -> pd.DataFrame
        Loads data from the CSV file.
    """

    def __init__(self, file_path: str):
        """
        Constructs all the necessary attributes for the DataLoader object.

        Parameters
        ----------
            file_path : str
                Path to the CSV file.
        """
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file.

        Returns
        -------
            pd.DataFrame
                Data loaded from the CSV file.
        """
        try:
            logger.info('Loading data from file: %s', self.file_path)
            return pd.read_csv(self.file_path)
        except Exception as e:
            logger.error('An error occurred while loading data from file. Error: %s', e)

# Create an instance of the DataLoader class with your file path
data_loader = DataLoader(r'Project\finalproject_itesm_mlops\data\fraud_oracle.csv')

# Load data from the CSV file
data = data_loader.load_data()
