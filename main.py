import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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


class Config:
    """Loads configuration from a YAML file or environment variables."""
    config_path = "config.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    else:
        logging.error(f"Configuration file '{config_path}' not found.")
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    
    DATA_PATH = config.get('DATA_PATH', './data/USA_Housing.csv')
    MODEL_SAVE_PATH = config.get('MODEL_SAVE_PATH', 'best_lasso_model.pkl')
    N_COMPONENTS = config.get('N_COMPONENTS', 5)
    TEST_SIZE = config.get('TEST_SIZE', 0.2)
    RANDOM_STATE = config.get('RANDOM_STATE', 12)

class ModelPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        setup_logging()

    def load_data(self) -> pd.DataFrame:
        """Loads and validates data from a specified CSV file."""
        try:
            df = pd.read_csv(self.config.DATA_PATH)
            logging.info(f"Data successfully loaded from {self.config.DATA_PATH}")
            return df
        except FileNotFoundError as e:
            logging.error(f'File not found: {self.config.DATA_PATH}. Please check the file path and try again.')
            raise e
        except pd.errors.EmptyDataError as e:
            logging.error(f'No data found in file: {self.config.DATA_PATH}. Please ensure the file is not empty.')
            raise e
        except pd.errors.ParserError as e:
            logging.error(f"Parsing error occurred while reading {self.config.DATA_PATH}. Check the file format.")
            raise e
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise e

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs feature engineering on the DataFrame."""
        if 'Avg. Area Number of Rooms' in df.columns and 'Avg. Area Number of Bedrooms' in df.columns:
            df['RoomBedroom_Ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']
            logging.info("Feature RoomBedroom_Ratio created.")
        return df

    def prepare_data(self, df: pd.DataFrame, target: str = 'Price'):
        """Prepares data by splitting it into training and test sets."""
        X = df.drop([target, 'Address'], axis=1, errors='ignore')
        y = df[target]
        return train_test_split(X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE)

    def create_preprocessing_pipeline(self) -> Pipeline:
        """Creates a preprocessing pipeline."""
        steps = [
            ('imputer', IterativeImputer(random_state=12)), 
            ('robust_scaler', RobustScaler())
        ]
        
        if self.config.N_COMPONENTS:
            steps.append(('pca', PCA(n_components=self.config.N_COMPONENTS)))
            logging.info(f"PCA will be applied with {self.config.N_COMPONENTS} components.")
        
        return Pipeline(steps=steps)

    def create_model_pipeline(self) -> Pipeline:
        """Creates a modeling pipeline that includes preprocessing."""
        preprocess_pipeline = self.create_preprocessing_pipeline()
        self.model = Pipeline(steps=[
            ('preprocess', preprocess_pipeline),
            ('model', Lasso(random_state=self.config.RANDOM_STATE))
        ])
        return self.model

    
    def hyperparameter_tuning_with_optuna(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Performs hyperparameter tuning using Optuna."""
        
        def objective(trial):
            alpha = trial.suggest_float('model__alpha', 0.01, 100, log=True)
            max_iter = trial.suggest_int('model__max_iter', 1000, 5000)
            
            pipeline.set_params(model__alpha=alpha, model__max_iter=max_iter)
            
            score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
            return score.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, timeout=600)
        
        logging.info(f"Best params: {study.best_params}")
        best_params = study.best_params
        
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        
        return pipeline

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluates the model performance on the test set."""
        y_pred = self.model.predict(X_test)
        eval_metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        logging.info(f"Model Evaluation: {eval_metrics}")
        return eval_metrics

    def save_model(self) -> None:
        """Serializes the model to a file."""
        try:
            joblib.dump(self.model, self.config.MODEL_SAVE_PATH)
            logging.info(f"Model saved to {self.config.MODEL_SAVE_PATH}")
        except Exception as e:
            logging.error(f"Failed to save the model: {e}")
            raise e

    def run(self):
        """Executes the end-to-end ML pipeline."""
        df = self.load_data()
        df = self.feature_engineering(df)
        
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        self.create_model_pipeline()
        
        best_model = self.hyperparameter_tuning_with_optuna(self.model, X_train, y_train)
        
        eval_metrics_best_model = self.evaluate_model(X_test, y_test)
        
        self.save_model()
        
        print("Lasso Best Model Metrics:")
        for metric, value in eval_metrics_best_model.items():
            print(f"    - {metric}: {value:,.4f}")

def setup_logging(log_file: str = "app.log"):
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

if __name__ == "__main__":
    pipeline = ModelPipeline(Config)
    pipeline.run()
