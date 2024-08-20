import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging

# Logging Conguration
logging.basicConfig(level=logging.INFO, format='%(astime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data Loaded Successfuly from {file_path}")
        return df
    except FileNotFoundError as e:
        logging.error(f'No file found in file: {file_path}')
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f'No data found in file: {file_path}')
        raise e
    except Exception as e:
        logging.error(f"An error occured while loading the data: {e}")

def handle_outliers(df, columns):
    """Handle oultiers using the 1st quartile and 3rd quartile"""
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

def feature_engineering(df):
    df['RoomBedroom_Ratio'] = df['Avg. Area Numbre of Rooms'] / df['Avg. Area Number of Bedrooms']
    return df

def preprocessing(df, n_components=5):
    """Prepare the data for training, applie PCA directly"""
    X = df.drop(['Price', 'Address'], axis=1, errors='ignore')
    y = df['Price']
    
    # Define dict witht the old names as key and kew shoter names as value
    rename_col = {
        'Avg. Area Income': 'Avg_Income',
        'Avg. Area House Age': 'House_Age',
        'Avg. Area Number of Rooms': 'Avg_Rooms',
        'Avg. Area Number of Bedrooms': 'Avg_Bedrooms',
        'Area Population': 'Population',
        'Price': 'Price',
        'RoomBedroom_Ratio': 'RoomBedroom_Ratio'
    }
    # Rename the columns using the rename() method
    X.rename(columns=rename_col, inplace=True)
    
    # Standardize the feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_processed = pca.fit_transform(X_scaled)
    return X_processed, y
    
    