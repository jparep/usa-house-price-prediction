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
    return X, y, poly, X_ploy

def train_lasso(X_train, y_train, X, poly, alpha=0.1):
    """Train a Lasso Regression Model"""
    lasso = Lasso(alpha=alpha, random_state=123)
    lasso.fit(X_train, y_train)
    
    lasso_coef = pd.DataFrame({
        'Feature': poly.get_feature_names_out(input_feature=X.columns),
        'Coefficient': lasso.coef_
    })
    logging.info('Lasso Model training Completed!')
    return lasso, lasso_coef

def train_linear_regression(X_train, y_train, poly, X_columns):
    """Trains a Linear Regression mdoel and performs cross-validation."""
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    cv_score = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    mean_cv_mae = -np.mean(cv_score)
    
    lr_coef = pd.DataFrame({
        'Feature': poly.get_feature_names_out(input_features=X_columns),
        'Coefficients': lr.coef_
    })
    logging.info('Linear Regression Trainng COmplete!')
    return lr, lr_coef, mean_cv_mae

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    y_pred = model.predict(X_test)
    
    evaluation_metrix = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    logging.info('Model Evaluation completed!')
    return evaluation_metrix


