import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

def load_data(file_path):
    """Loads data from a specified CSV file path."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data successfully loaded from {file_path}")
        return df
    except FileNotFoundError as e:
        logging.error(f'File not found: {file_path}')
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f'No data found in file: {file_path}')
        raise e
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise e

def handle_outliers(df, columns):
    """Handles outliers using the 1st and 3rd quartiles (IQR method)."""
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
            logging.info(f"Outliers handled for column: {col}")
    return df

def feature_engineering(df):
    """Performs feature engineering on the DataFrame."""
    if 'Avg. Area Number of Rooms' in df.columns and 'Avg. Area Number of Bedrooms' in df.columns:
        df['RoomBedroom_Ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']
        logging.info("Feature RoomBedroom_Ratio created.")
    return df

def preprocess_data(df, n_components=None):
    """Prepares the data for training, optionally applies PCA."""
    X = df.drop(['Price', 'Address'], axis=1, errors='ignore')
    y = df['Price']
    
    # Standardize the feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA if requested
    if n_components:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        X_scaled = pca.fit_transform(X_scaled)
        logging.info(f"PCA applied with {n_components} components.")
    
    return X_scaled, y

def train_model(model, X_train, y_train):
    """Trains a model with cross-validation and returns the model."""
    cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=['neg_mean_absolute_error', 'r2'])
    mean_cv_mae = -cv_results['test_neg_mean_absolute_error'].mean()
    mean_cv_r2 = cv_results['test_r2'].mean()
    logging.info(f"Trained {model.__class__.__name__} with CV MAE: {mean_cv_mae:.4f}, R2: {mean_cv_r2:.4f}")
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(model, X_train, y_train):
    """Implement hyperparameter tunning for the model"""
    param_grid = {
        
    }
    cv = GridSearchCV()
    best_model = cv.best_estimator_
    bet_prama = cv.best_params_
    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model performance on the test set."""
    y_pred = model.predict(X_test)
    
    eval_metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    logging.info(f"Model Evaluation: {eval_metrics}")
    return eval_metrics

def main():
    # Load Data
    file_path = './data/USA_Housing.csv'
    df = load_data(file_path)
    
    # Handle Outliers
    outlier_columns = ['Avg. Area Income', 'Avg. Area House Age']
    df = handle_outliers(df, outlier_columns)
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Preprocessing Data
    X_processed, y = preprocess_data(df)
    
    # Split Data into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=12)
    
    # Define Lasso Regression Model
    lasso_model = Lasso(alpha=0.1, random_state=12)
    
    # Train model
    trained_model = train_model(lasso_model, X_train, y_train)
    
    # Evaluate the Model
    eval_metrics = evaluate_model(trained_model, X_test, y_test)
    
    # Print Model Evaluation Metrics
    print("Lasso Model Metrics:")
    for metric, value in eval_metrics.items():
        print(f"    - {metric}: {value:,.4f}")

if __name__ == "__main__":
    main()
