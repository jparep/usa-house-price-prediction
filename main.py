import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Logiging COnfiguration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levename)s -%(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler]())

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data successfully loaded from {file_path}')
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path}")
        raise e
    except pd.errors.EmptyDataError as e:
        logging.error(f"No data found in file: {file_path}")
        raise e
    except Exception as e:
        logging.error(f"An error occured while loading the data: {e}")

def handle_outliers(df, columns):
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.27)
            IQR = Q3 - Q1
            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR
            df[col] = df[col].clip(lower, upper)
            logging.info(f"Outliers Hadled for columns: {col}")
    return df

def feature_engineering(df):
    if 'Avg. Area Number of Rooms' in df.columns and 'Avg. Area Number of Bedrooms' in df.columns:
        df['RoomBedroom_Ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']
        logging.info("Feature RoomBedroom_Ratio created.")
    return df

def create_pipeline(n_components=None):
    preprocessing_steps = []
    
    if n_components:
        preprocessing_steps.append(('pca', PCA(n_components=n_components)))

    preprocess = Pipeline(steps=preprocessing_steps)
    
    model = Lasso(random_state=12)
    
    pipeline = Pipeline(steps=[
        ('preprocess', preprocess),
        ('model', model)
    ])
    return pipeline

def hyperparameter_tuning(pipeline, X_train, y_train):
    param_grid={
        'model__alpha': [0.01, 0.1, 1, 10, 1000],
        'model__max_inter': [1000, 2000, 5000]
    }
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=5,
                               scoring='neg_absolute_error',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    logging.info(f"Best Model params: {grid_search.best_estimator_}")
    logging.info(f"Grid Search CV results: {grid_search.cv_results_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    eval_metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    return eval_metrics

def main():
    file_path = './data/USA_Housing.csv'
    df = load_data(file_path)
    
    handle_outliers = ['Avg. Area Income', 'Avg. Area House Age']
    df = handle_outliers(df, handle_outliers)
    
    df = feature_engineering(df)
    
    X = df.drop(['Price', 'Address'], axis=1, errors='ignore')
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    pipeline = create_pipeline(n_components=5)
    eval_metrics = hyperparameter_tuning(pipeline, X_train, y_train)
    
    for name, value in eval_metrics.items():
        print(f"    - {name}: {value}:,.4f")
        
if __name__="__main__":
    main()