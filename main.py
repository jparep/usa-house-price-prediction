import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Loads housing data from a CSV file into a DataFrame."""
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

def preprocess_data(df, apply_pca=False, n_components=None, poly_degree=2):
    """Prepares the feature matrix and target vector for model training."""
    X = df.drop(['Price', 'Address'], axis=1, errors='ignore')
    y = df['Price']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if apply_pca and n_components:
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        logging.info(f'PCA applied with {n_components} components.')
        transformer = pca
    else:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_reduced = poly.fit_transform(X_scaled)
        logging.info(f'Polynomial features generated with degree {poly_degree}.')
        transformer = poly
    
    logging.info('Preprocessing data complete!')
    return X, y, X_reduced, transformer


def train_lasso(X_train, y_train, X_columns, transformer, alpha=0.1):
    """Trains a Lasso Regression model."""
    lasso = Lasso(alpha=alpha, random_state=123)
    lasso.fit(X_train, y_train)
    
    lasso_coef = pd.DataFrame({
        'Feature': transformer.get_feature_names_out(input_features=X_columns) if isinstance(transformer, PolynomialFeatures) else [f'PC{i+1}' for i in range(transformer.n_components_)],
        'Coefficient': lasso.coef_
    })
    logging.info('Lasso model training completed!')
    return lasso, lasso_coef

def train_linear_regression(X_train, y_train, transformer, X_columns):
    """Trains a Linear Regression model and performs cross-validation."""
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    cv_score = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    mean_cv_mae = -np.mean(cv_score)
    
    lr_coef = pd.DataFrame({
        'Feature': transformer.get_feature_names_out(input_features=X_columns) if isinstance(transformer, PolynomialFeatures) else [f'PC{i+1}' for i in range(transformer.n_components_)],
        'Coefficient': lr.coef_
    })
    logging.info('Linear Regression training completed!')
    return lr, lr_coef, mean_cv_mae

def train_ridge(X_train, y_train, X_columns, transformer, alpha=1.0):
    """Trains a Ridge Regression model."""
    ridge = Ridge(alpha=alpha, random_state=123)
    ridge.fit(X_train, y_train)
    
    ridge_coef = pd.DataFrame({
        'Feature': transformer.get_feature_names_out(input_features=X_columns) if isinstance(transformer, PolynomialFeatures) else [f'PC{i+1}' for i in range(transformer.n_components_)],
        'Coefficient': ridge.coef_
    })
    logging.info('Ridge model training completed!')
    return ridge, ridge_coef

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set."""
    y_pred = model.predict(X_test)
    
    evaluation_metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    logging.info('Model evaluation completed!')
    return evaluation_metrics

def select_best_model(lasso_metrics, lr_metrics, ridge_metrics):
    """Selects the best model based on the evaluation metrics."""
    best_model = 'Lasso'
    best_metrics = lasso_metrics
    
    if lr_metrics['MAE'] < best_metrics['MAE']:
        best_model = 'Linear Regression'
        best_metrics = lr_metrics
    
    if ridge_metrics['MAE'] < best_metrics['MAE']:
        best_model = 'Ridge Regression'
        best_metrics = ridge_metrics
    
    logging.info(f'Best model selected: {best_model}')
    return best_model, best_metrics

if __name__ == "__main__":
    file_path = './data/USA_Housing.csv'
    df = load_data(file_path)
    df = feature_engineering(df)
    
    # Option to apply PCA or Polynomial Features
    use_pca = False
    n_components = 5 if use_pca else None
    
    X, y, X_reduced, transformer = preprocess_data(df, apply_pca=use_pca, n_components=n_components)
    
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=123)
    
    # Train Lasso Model
    lasso, lasso_coef = train_lasso(X_train, y_train, X.columns, transformer, alpha=0.1)
    lasso_metrics = evaluate_model(lasso, X_test, y_test)
    print(f"Lasso Model Evaluation Metrics: {lasso_metrics}")
    
    # Train Linear Regression model
    lr, lr_coef, mean_cv_mae = train_linear_regression(X_train, y_train, transformer, X.columns)
    lr_metrics = evaluate_model(lr, X_test, y_test)
    print(f'Linear Regression Model Evaluation Metrics: {lr_metrics}')
    
    # Train Ridge Regression model
    ridge, ridge_coef = train_ridge(X_train, y_train, X.columns, transformer, alpha=1.0)
    ridge_metrics = evaluate_model(ridge, X_test, y_test)
    print(f'Ridge Regression Model Evaluation Metrics: {ridge_metrics}')
    
    # Select and output the best model
    best_model, best_metrics = select_best_model(lasso_metrics, lr_metrics, ridge_metrics)
    print(f"The best model is: {best_model}")
    
    if best_model == 'Lasso':
        print(f"Lasso Coefficients: \n{lasso_coef}")
    elif best_model == 'Linear Regression':
        print(f'Linear Regression Coefficients:\n{lr_coef}')
    else:
        print(f'Ridge Regression Coefficients:\n{ridge_coef}')
