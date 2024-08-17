import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the housing data into DataFrame
df = pd.read_csv('./data/USA_Housing.csv')

# Feature Engineering: Adding Room-to-Bedroom Ration as a new features
df['Room2Bedroom_ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']

# Define the feature variables and target variable
X = df.drop(['Price', 'Address'])
y = df['Price']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

