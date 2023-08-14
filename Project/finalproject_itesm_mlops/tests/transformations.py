import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Test the custom label encoder from the fraud-detection notebook
class CustomLabelEncoder:
    def __init__(self, col_ordering):
        self.col_ordering = col_ordering
        self.label_encoders = {}

    def fit_transform(self, df):
        for item in self.col_ordering:
            col = item['col']
            mapping = item['mapping']
            le = LabelEncoder()
            le.fit_transform(list(mapping.keys()))
            self.label_encoders[col] = le

            df[col] = df[col].map(mapping)

        return df

    def transform(self, df):
        for col, le in self.label_encoders.items():
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else None)

        return df

# Create a sample DataFrame
data = {
    'col1': ['apple', 'orange', 'banana', 'orange'],
    'col2': ['red', 'green', 'yellow', 'green']
}
df = pd.DataFrame(data)

# Define the column ordering and corresponding mappings
col_ordering = [
    {'col': 'col1', 'mapping': {'apple': 1, 'banana': 2, 'orange': 3}},
    {'col': 'col2', 'mapping': {'red': 4, 'green': 5, 'yellow': 6}}
]

# Create an instance of CustomLabelEncoder
encoder = CustomLabelEncoder(col_ordering)

# Fit and transform the DataFrame using the CustomLabelEncoder
transformed_df = encoder.fit_transform(df)

# Check if the transformation is valid
is_valid = all(transformed_df[col].notna().all() for col in transformed_df.columns)

# Print the result
if is_valid:
    print("The transformation is valid.")
else:
    print("The transformation is not valid.")
