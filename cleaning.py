import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Loading the dataset
data = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# Checking for missing values
print(data.isnull().sum())

# Fill missing values with the mean of the column
data.fillna(data.mean(), inplace=True)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Assuming 'sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', and 'ST slope' are categorical
categorical_columns = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope']

# Converting categorical columns to numerical format using one-hot encoding
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Separating features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)