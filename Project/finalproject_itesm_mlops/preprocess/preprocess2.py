import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from custom_transformers import DropColumnsTransformer, FillNaTransformer, OneHotEncodingTransformer, StandardScalerTransformer
import os
import logging

log_filename = r'C:\Users\brianda.nunez\Documents\GitHub\finalproject\Project\finalproject_itesm_mlops\preprocess\preprocess.log'
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info("Preprocessing data...")

        drop_columns_transformer = DropColumnsTransformer(columns=['PolicyNumber', 'PolicyType'])
        #fill_na_transformer = FillNaTransformer(columns=['card6'])
        one_hot_encoding_transformer = OneHotEncodingTransformer(columns=['AccidentArea', 'Sex', 'Fault', 'PoliceReportFiled', 'WitnessPresent','AgentType','Month','DayOfWeek','DayOfWeekClaimed','MonthClaimed','PastNumberOfClaims','NumberOfSuppliments','VehiclePrice','AgeOfVehicle','Days_Policy_Accident','Days_Policy_Claim','AgeOfPolicyHolder','AddressChange_Claim','NumberOfCars'])
        #standard_scaler_transformer = StandardScalerTransformer(columns=['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2'])

        preprocessed_data = drop_columns_transformer.transform(data)
        #preprocessed_data = fill_na_transformer.transform(preprocessed_data)
        preprocessed_data = one_hot_encoding_transformer.transform(preprocessed_data)
        #preprocessed_data = standard_scaler_transformer.transform(preprocessed_data)

        return preprocessed_data


logging.info("Loading fraud data...")
data = pd.read_csv(r'Project\finalproject_itesm_mlops\data\fraud_oracle.csv')

logging.info("Splitting features and labels...")
X = data.drop("FraudFound_P", axis=1)
y = data["FraudFound_P"]

preprocessor = DataPreprocessor()
preprocessed_data = preprocessor.preprocess(data)

output_file = os.path.join(r'Project\finalproject_itesm_mlops\preprocess', 'output_file.csv')

output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logging.info("Saving preprocessed data to CSV: %s", output_file)
preprocessed_data.to_csv(output_file, index=False)

logging.info("Splitting preprocessed data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, y, test_size=0.2, random_state=42)

logging.info("Training fraud classifier...")
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

logging.info("Accuracy: %f", accuracy)