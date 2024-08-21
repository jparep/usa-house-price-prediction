import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import joblib
import logging
import optuna
from optuna.integration import SklearnPruningCallback

# Logging Configuration
def setup_logging(log_file: str = "app.log"):
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Configuration Management
class Config:
    """Loads configuration from a YAML file or environment variables."""
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    DATA_PATH = config.get('DATA_PATH', './data/USA_Housing.csv')
    MODEL_SAVE_PATH = config.get('MODEL_SAVE_PATH', 'best_lasso_model.pkl')
    N_COMPONENTS = config.get('N_COMPONENTS', 5)
    TEST_SIZE = config.get('TEST_SIZE', 0.2)
    RANDOM_STATE = config.get('RANDOM_STATE', 12)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads and validates data from a specified CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data successfully loaded from {file_path}")
        return df
    except FileNotFoundError as e:
        logging.error(f'File not found: {file_path}. Please check the file path and try again.')
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f'No data found in file: {file_path}. Please ensure the file is not empty.')
        raise e
    except pd.errors.ParserError as e:
        logging.error(f"Parsing error occurred while reading {file_path}. Check the file format.")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
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

def hyperparameter_tuning_with_optuna(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Performs hyperparameter tuning using Optuna."""
    def objective(trial):
        alpha = trial.suggest_loguniform('alpha', 1e-3, 1e1)
        max_iter = trial.suggest_int('max_iter', 1000, 5000)
        
        pipeline.set_params(model__alpha=alpha, model__max_iter=max_iter)
        
        pruning_callback = SklearnPruningCallback(trial, "neg_mean_absolute_error")
        score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
                                error_score='raise', fit_params={'callbacks': [pruning_callback]})
        return score.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=600)
    
    logging.info(f"Best params from Optuna: {study.best_params}")
    best_params = study.best_params
    
    pipeline.set_params(model__alpha=best_params['alpha'], model__max_iter=best_params['max_iter'])
    pipeline.fit(X_train, y_train)
    
    return pipeline

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

def main():
    """Executes the end-to-end ML pipeline with logging and configuration management."""
    setup_logging()
    
    df = load_data(Config.DATA_PATH)
    df = feature_engineering(df)
    
    X_train, X_test, y_train, y_test = prepare_data(df, target='Price', 
                                                    test_size=Config.TEST_SIZE, 
                                                    random_state=Config.RANDOM_STATE)
    
    preprocess_pipeline = create_preprocessing_pipeline(n_components=Config.N_COMPONENTS)
    pipeline = create_model_pipeline(preprocess_pipeline)
    
    best_model = hyperparameter_tuning_with_optuna(pipeline, X_train, y_train)
    
    evaluator = ModelEvaluator(best_model)
    eval_metrics_best_model = evaluator.evaluate(X_test, y_test)
    
    save_model(best_model, Config.MODEL_SAVE_PATH)
    
    print("Lasso Best Model Metrics:")
    for metric, value in eval_metrics_best_model.items():
        print(f"    - {metric}: {value:,.4f}")

if __name__ == "__main__":
    main()
