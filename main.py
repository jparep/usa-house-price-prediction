import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import joblib
import logging
import numpy as np

# Logging Configuration
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

def load_data(file_path):
    """Loads data from a specified CSV file path and validates it."""
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

def feature_engineering(df):
    """Performs feature engineering on the DataFrame."""
    if 'Avg. Area Number of Rooms' in df.columns and 'Avg. Area Number of Bedrooms' in df.columns:
        df['RoomBedroom_Ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']
        logging.info("Feature RoomBedroom_Ratio created.")
    return df

def create_pipeline(n_components=None):
    """Creates a pipeline for preprocessing and modeling."""
    # Define preprocessing steps
    preprocessing_steps = [
        ('imputer', IterativeImputer(random_state=12)),  # Handle missing data
        ('robust_scaler', RobustScaler())  # Robust Scaler to handle outliers
    ]
    
    # PCA if requested
    if n_components:
        preprocessing_steps.append(('pca', PCA(n_components=n_components)))
        logging.info(f"PCA will be applied with {n_components} components.")
    
    # Combine preprocessing steps into a Pipeline
    preprocess = Pipeline(steps=preprocessing_steps)
    
    # Define the model
    model = Lasso(random_state=12)
    
    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocess', preprocess),
        ('model', model)
    ])
    
    return pipeline

def hyperparameter_tuning(pipeline, X_train, y_train):
    """Performs hyperparameter tuning using GridSearchCV."""
    param_grid = {
        'model__alpha': [0.01, 0.1, 1, 10, 100],
        'model__max_iter': [1000, 2000, 5000]
    }
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best Model params: {grid_search.best_params_}")
    logging.info(f"Grid Search CV results: {grid_search.cv_results_}")
    
    return grid_search.best_estimator_

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

def save_model(model, file_path):
    """Serializes the model to a file."""
    try:
        joblib.dump(model, file_path)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save the model: {e}")
        raise e

def main():
    # Load Data
    file_path = './data/USA_Housing.csv'
    df = load_data(file_path)
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Prepare Data for Training
    X = df.drop(['Price', 'Address'], axis=1, errors='ignore')
    y = df['Price']
    
    # Split Data into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    # Create and tune the pipeline
    pipeline = create_pipeline(n_components=5)
    best_model = hyperparameter_tuning(pipeline, X_train, y_train)
    
    # Evaluate the Best Model
    eval_metrics_best_model = evaluate_model(best_model, X_test, y_test)
    
    # Save the Best Model
    save_model(best_model, 'best_lasso_model.pkl')
    
    # Print Best Model evaluation metrics
    print("Lasso Best Model Metrics:")
    for metric, value in eval_metrics_best_model.items():
        print(f"    - {metric}: {value:,.4f}")

if __name__ == "__main__":
    main()
