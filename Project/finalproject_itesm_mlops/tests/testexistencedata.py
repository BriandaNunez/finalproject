import unittest
import pandas as pd

class TestDataExistence(unittest.TestCase):
    def test_data_existence(self):
        # Load training data
        df_train = pd.read_csv(r'Project\finalproject_itesm_mlops\data\fraud_oracle.csv')

        # Verify training data existance
        self.assertFalse(df_train.empty, "No training data")

if __name__ == '__main__':
    unittest.main()
