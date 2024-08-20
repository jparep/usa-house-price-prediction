import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load the data from a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data loaded successfully from {file_path}')
        return df
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        raise

def handle_outliers(df, columns):
    """Handle outliers using the 1st quartile (Q1) and 3rd quartile (Q3) for the given columns."""
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.clip(df[column], lower_bound, upper_bound)
        logging.info(f'Outliers in {column} handled by capping at Q1 and Q3 with IQR method.')
    return df

def feature_engineering(df, handle_outliers=False):
    """Perform feature engineering on the dataset, with optional outlier handling."""
    df['Room2Bedroom_ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']   
    return df

def preprocess_data(df, n_components=5):
    """Prepare the data for model training, applies PCA directly."""
    X = df.drop(['Price', 'Address'], axis=1, errors='ignore')
    y = df['Price']
    
    # Standardize the feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_processed = pca.fit_transform(X_scaled)
    logging.info(f'PCA applied with {n_components} components.')
    
    return y, X_processed

def train_model(model, X_train, y_train):
    """Train a model with cross-validation."""
    model.fit(X_train, y_train)
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    mean_cv_mae = -np.mean(cv_score)
    logging.info(f'Trained {model.__class__.__name__} with CV MAE: {mean_cv_mae:.4f}')
    return model, mean_cv_mae

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data."""
    y_pred = model.predict(X_test)
    
    # Calculate the Evaluation Metrics
    eval_metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    return eval_metrics

if __name__ == "__main__":
    file_path = './data/USA_Housing.csv'
    
    # Load and process the data
    df = load_data(file_path)
    
    df = feature_engineering(df, handle_outliers)
    
    # Preprocess the data with PCA applied
    y, X_processed = preprocess_data(df)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=123)
    
    # Define the Lasso model
    lasso_model = Lasso(alpha=0.1, random_state=12)
    
    # Train and evaluate the Lasso model
    trained_model, _ = train_model(lasso_model, X_train, y_train)
    metrics = evaluate_model(trained_model, X_test, y_test)
    
    # Print the evaluation metrics
    print(f"Lasso Model Metrics:")
    print(f"  - MAE: {metrics['MAE']:,.2f}")
    print(f"  - MSE: {metrics['MSE']:,.2f}")
    print(f"  - R2 : {metrics['R2']*100:.2f}%")
