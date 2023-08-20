import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from custom_transformers import DropColumnsTransformer, FillNaTransformer, OneHotEncodingTransformer, StandardScalerTransformer
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create a logger for the preprocess module
logger = logging.getLogger('preprocess')
logger.setLevel(logging.INFO)

# Create a file handler for the preprocess logger
log_file = r'C:\Users\brianda.nunez\Documents\GitHub\finalproject\Project\finalproject_itesm_mlops\preprocess\preprocess.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the preprocess logger
logger.addHandler(file_handler)

class DataPreprocessor:
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info('Starting data preprocessing')

        drop_columns_transformer = DropColumnsTransformer(columns=['PolicyNumber', 'PolicyType'])
        one_hot_encoding_transformer = OneHotEncodingTransformer(columns=['AccidentArea', 'Sex', 'Fault', 'PoliceReportFiled', 'WitnessPresent','AgentType','Month','DayOfWeek','DayOfWeekClaimed','MonthClaimed','PastNumberOfClaims','NumberOfSuppliments','VehiclePrice','AgeOfVehicle','Days_Policy_Accident','Days_Policy_Claim','AgeOfPolicyHolder','AddressChange_Claim','NumberOfCars'])
        
        preprocessed_data = drop_columns_transformer.transform(data)
        preprocessed_data = one_hot_encoding_transformer.transform(preprocessed_data)
        
        logger.info('Data preprocessing completed')
        
        return preprocessed_data


# Cargar datos de fraude
data = pd.read_csv(r'Project\finalproject_itesm_mlops\data\fraud_oracle.csv')

# Dividir características y etiquetas
X = data.drop("FraudFound_P", axis=1)
y = data["FraudFound_P"]

# Preprocesar los datos
preprocessor = DataPreprocessor()
preprocessed_data = preprocessor.preprocess(data)

# Especificar la ruta y el nombre del nuevo archivo CSV
output_file = os.path.join(r'C:\Users\brianda.nunez\Documents\GitHub\finalproject\Project\finalproject_itesm_mlops\preprocess', 'output_file.csv')

# Comprobar si el directorio para guardar el archivo existe. Si no existe, crearlo.
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Guardar los datos preprocesados en el nuevo archivo CSV
preprocessed_data.to_csv(output_file, index=False)

# Registrar el nombre y la ruta del archivo
logger.info(f"Datos preprocesados guardados en: {output_file}")

# Dividir datos preprocesados en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(preprocessed_data, y, test_size=0.2, random_state=42)

# Entrenar y evaluar el clasificador de fraude
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)

# Registrar la precisión
logger.info(f"Precisión: {accuracy}")
print("Precisión:", accuracy)
