# Import necessary modules
import data_preprocessor as dp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the dataset
messy_data = pd.read_csv('messy_data.csv')
# print(messy_data.shape)
clean_data = messy_data.copy()

# 2. Preprocess the data
clean_data = dp.impute_missing_values(clean_data, strategy='mean')
# print("------------------1",clean_data.shape)
clean_data = dp.remove_duplicates(clean_data)
# print("------------------2",clean_data.shape)
clean_data = dp.normalize_data(clean_data)
# print("------------------3",clean_data.info())
clean_data = dp.remove_redundant_features(clean_data)

# print(clean_data.head())
# print(clean_data.info())
# print(clean_data.describe())

# 3. Save the cleaned dataset
clean_data.to_csv('clean_data.csv', index=False)

# 4. Train and evaluate the model
dp.simple_model(clean_data)
