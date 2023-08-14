import pandas as pd

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
        return pd.read_csv(self.file_path)

# Create an instance of the DataLoader class with your file path
data_loader = DataLoader(r'Project\finalproject_itesm_mlops\data\fraud_oracle.csv')

# Load data from the CSV file
data = data_loader.load_data()
