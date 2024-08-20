import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
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

