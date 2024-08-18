import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging


# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load the data from CSV file into DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data loaded successfuly from {file_path}')
        return df
    except Exception as e:
        logging.info(f'Error loading data {e}')
        raise

def feature_engineering(df):
    """Perform feature engineering on the dataset."""
    df['Room2BedRoom_ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']
    return df

def preprocess_data(df, n_components=5):
    """Prepare the data for mdoel trainng, applies PCA directly."""
    X = df.drop(['Price', 'Address'], axis=1, errors='ignore')
    y = df['Price']
    
    # Standardize the feature metrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_processed = pca.fit_transform(X_scaled)
    return X_processed, pca

def lasso_training(X_train, y_train, alpha=0.1):
    """Train Lasso REgression Model"""
    l1 = Lasso(alpha=alpha, random_state=12)
    l1.fit(X_train, y_train)
    return l1

def ridge_training(X_train, y_train, alpha=0.1):
    """Train Ridge Regression model"""
    l2 = Ridge(alpha=alpha, random_state=12)
    l2.fit(X_train, y_train)
    return l2

def linear_regression_training(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Perform cross-validation to evaluate model performance
    cv_score = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    mean_cv_mae = -np.mean(cv_score)
    
    return lr, mean_cv_mae
    
def elastic_net_training(X_train, y_train, alpha=1.0, l1_ratio=0.5):
    """Train an ElasticNet Model"""
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12)
    en.fit(X_train, y_train)
    return en

def random_forest_training(X_train, y_train, n_estimators =100, random_state=12):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf

def svr_training(X_train, y_train, kernel='rbf', C=1.0):
    svr = SVR(kernel=kernel, C=C)
    svr.fit(X_train, y_train)
    return svr  

def gradient_boost_training(X_train, y_train, n_estimator=100, learning_rate=0.1, random_state=123):
    gb = GradientBoostingRegressor(n_estimator=n_estimator, learning_rate=learning_rate, random_state=random_state)
    gb.fit(X_train, y_train)
    return gb

def xgboost_training(X_train, y_train, n_estimators=100, learning_rate=0.1, random_state=12):
    xgb = XGBRFRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    xgb.fit(X_train , y_train)
    return xgb


