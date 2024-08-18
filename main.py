import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """ Loads housing data from a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data loaded successfully from {file_path}')
        return df
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        raise
    
def feature_engineering(df):
    """Performs feature engineering on the dataset."""
    df['Room2Bedroom_ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']
    logging.info('Feature engineering completed!')
    return df

def preprocess_data(df):
    """Prepares the featuers matrix and target vector for model training"""
    X = df.drop(['Price', 'Address'], axis=1, errors='ignore')
    y = df['Price']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_ploy = poly.fit_transform(X_scaled)
    
    logging.info('Preprocessing data complete!')
    return X_ploy, poly
