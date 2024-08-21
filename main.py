import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import joblib
import logging

# Logging Configuration
def setup_logging(log_file: str = "app.log"):
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

class Config:
    DATA_PATH = './data/USA_Housing.csv'
    MODEL_SAVE_PATH = 'best_lasso_model.pkl'
    N_COMPONENTS = 5
    TEST_SIZE = 0.2
    RANDOM_STATE = 12

def load_data(file_path: str) -> pd.DataFrame:
    """Loads and validates data from a specified CSV file."""
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

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Performs feature engineering on the DataFrame."""
    if 'Avg. Area Number of Rooms' in df.columns and 'Avg. Area Number of Bedrooms' in df.columns:
        df['RoomBedroom_Ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']
        logging.info("Feature RoomBedroom_Ratio created.")
    return df

def prepare_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 12):
    """Prepares data by splitting it into training and test sets."""
    X = df.drop([target, 'Address'], axis=1, errors='ignore')
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_preprocessing_pipeline(n_components: int = None) -> Pipeline:
    """Creates a preprocessing pipeline."""
    steps = [
        ('imputer', IterativeImputer(random_state=12)), 
        ('robust_scaler', RobustScaler())
    ]
    
    if n_components:
        steps.append(('pca', PCA(n_components=n_components)))
        logging.info(f"PCA will be applied with {n_components} components.")
    
    return Pipeline(steps=steps)

def create_model_pipeline(preprocess_pipeline: Pipeline, model=Lasso(random_state=12)) -> Pipeline:
    """Creates a modeling pipeline that includes preprocessing."""
    return Pipeline(steps=[
        ('preprocess', preprocess_pipeline),
        ('model', model)
    ])

def hyperparameter_tuning(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Performs hyperparameter tuning using GridSearchCV."""
    param_grid = {
        'model__alpha': [0.01, 0.1, 1, 10, 100],
        'model__max_iter': [1000, 2000, 5000]
    }
    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best Model params: {grid_search.best_params_}")
    logging.info(f"Grid Search CV results: {grid_search.cv_results_}")
    
    return grid_search.best_estimator_

class ModelEvaluator:
    def __init__(self, model: Pipeline):
        self.model = model
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluates the model performance on the test set."""
        y_pred = self.model.predict(X_test)
        eval_metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        logging.info(f"Model Evaluation: {eval_metrics}")
        return eval_metrics

def save_model(model: Pipeline, file_path: str) -> None:
    """Serializes the model to a file."""
    try:
        joblib.dump(model, file_path)
        logging.info(f"Model saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save the model: {e}")
        raise e

def main(file_path: str = Config.DATA_PATH, model_save_path: str = Config.MODEL_SAVE_PATH):
    """Executes data loading, preprocessing, model training, tuning, evaluation, and saving."""
    setup_logging()
    
    df = load_data(file_path)
    df = feature_engineering(df)
    
    X_train, X_test, y_train, y_test = prepare_data(df, target='Price', 
                                                    test_size=Config.TEST_SIZE, 
                                                    random_state=Config.RANDOM_STATE)
    
    preprocess_pipeline = create_preprocessing_pipeline(n_components=Config.N_COMPONENTS)
    pipeline = create_model_pipeline(preprocess_pipeline)
    
    best_model = hyperparameter_tuning(pipeline, X_train, y_train)
    
    evaluator = ModelEvaluator(best_model)
    eval_metrics_best_model = evaluator.evaluate(X_test, y_test)
    
    save_model(best_model, model_save_path)
    
    print("Lasso Best Model Metrics:")
    for metric, value in eval_metrics_best_model.items():
        print(f"    - {metric}: {value:,.4f}")

if __name__ == "__main__":
    main()
