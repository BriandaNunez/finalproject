import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from custom_transformers import DropColumnsTransformer, FillNaTransformer, OneHotEncodingTransformer, StandardScalerTransformer
import os

class DataPreprocessor:
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        drop_columns_transformer = DropColumnsTransformer(columns=['PolicyNumber', 'PolicyType'])
        #fill_na_transformer = FillNaTransformer(columns=['card6'])
        one_hot_encoding_transformer = OneHotEncodingTransformer(columns=['AccidentArea', 'Sex', 'Fault', 'PoliceReportFiled', 'WitnessPresent','AgentType','Month','DayOfWeek','DayOfWeekClaimed','MonthClaimed','PastNumberOfClaims','NumberOfSuppliments','VehiclePrice','AgeOfVehicle','Days_Policy_Accident','Days_Policy_Claim','AgeOfPolicyHolder','AddressChange_Claim','NumberOfCars'])
        #standard_scaler_transformer = StandardScalerTransformer(columns=['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2'])

        preprocessed_data = drop_columns_transformer.transform(data)
        #preprocessed_data = fill_na_transformer.transform(preprocessed_data)
        preprocessed_data = one_hot_encoding_transformer.transform(preprocessed_data)
        #preprocessed_data = standard_scaler_transformer.transform(preprocessed_data)

        return preprocessed_data


# Load fraud data
data = pd.read_csv(r'Project\finalproject_itesm_mlops\data\fraud_oracle.csv')

# Split features and labels
X = data.drop("FraudFound_P", axis=1)
y = data["FraudFound_P"]

# Preprocess the data
preprocessor = DataPreprocessor()
preprocessed_data = preprocessor.preprocess(data)

# Specify the file path and name for the new CSV file
output_file = os.path.join(r'Project\finalproject_itesm_mlops\preprocess', 'output_file.csv')


# Check if the directory to save the file exists. If not, create it.
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the preprocessed data into the new CSV file
preprocessed_data.to_csv(output_file, index=False)

# Split preprocessed data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, y, test_size=0.2, random_state=42)

# Train and evaluate fraud classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)