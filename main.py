import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load the data from CSV file into DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Data loaded successfully from {file_path}')
        return df
    except Exception as e:
        logging.error(f'Error loading data {e}')
        raise

def feature_engineering(df):
    """Perform feature engineering on the dataset."""
    df['Room2Bedroom_ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']
    
    # Filter out non-numeric columns before skewness correction
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Handle skewness in numeric features
    skewed_feats = numeric_df.apply(lambda x: x.skew()).sort_values(ascending=False)
    skewness_threshold = 0.75
    skewed_features = skewed_feats[abs(skewed_feats) > skewness_threshold].index
    
    if len(skewed_features) > 0:
        pt = PowerTransformer(method='yeo-johnson')
        try:
            df[skewed_features] = pt.fit_transform(df[skewed_features])
            logging.info('Skewness correction applied to features.')
        except ValueError as e:
            logging.error(f'Error applying PowerTransformer: {e}')
            raise
    else:
        logging.info('No skewed features found for transformation.')
    
    logging.info('Feature engineering and skewness correction completed!')
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

def lasso_training(X_train, y_train, alpha=0.1):
    """Train Lasso Regression Model."""
    l1 = Lasso(alpha=alpha, random_state=12)
    l1.fit(X_train, y_train)
    return l1

def ridge_training(X_train, y_train, alpha=0.1):
    """Train Ridge Regression model."""
    l2 = Ridge(alpha=alpha, random_state=12)
    l2.fit(X_train, y_train)
    return l2

def linear_regression_training(X_train, y_train):
    """Train Linear Regression model."""
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Perform cross-validation to evaluate model performance
    cv_score = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    mean_cv_mae = -np.mean(cv_score)
    
    return lr, mean_cv_mae
    
def elastic_net_training(X_train, y_train, alpha=1.0, l1_ratio=0.5):
    """Train an ElasticNet model."""
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12)
    en.fit(X_train, y_train)
    return en

def random_forest_training(X_train, y_train, n_estimators=100, random_state=12):
    """Train a Random Forest Regressor model."""
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf

def svr_training(X_train, y_train, kernel='rbf', C=1.0):
    """Train a Support Vector Regressor model."""
    svr = SVR(kernel=kernel, C=C)
    svr.fit(X_train, y_train)
    return svr  

def gradient_boost_training(X_train, y_train, n_estimators=100, learning_rate=0.1, random_state=123):
    """Train a Gradient Boosting Regressor model."""
    gb = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    gb.fit(X_train, y_train)
    return gb

def xgboost_training(X_train, y_train, n_estimators=100, learning_rate=0.1, random_state=12):
    """Train an XGBoost Regressor model."""
    xgb = XGBRFRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    xgb.fit(X_train, y_train)
    return xgb

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
    
def select_best_model(eval_metrics):
    """Select the best model based on MAE."""
    best_model = min(eval_metrics, key=lambda k: eval_metrics[k]['MAE'])
    best_metrics = eval_metrics[best_model]
    return best_model, best_metrics

if __name__ == "__main__":
    file_path = './data/USA_Housing.csv'
    
    # Load and process the data
    df = load_data(file_path)
    df = feature_engineering(df)
    
    # Preprocess the data with PCA applied
    y, X_processed = preprocess_data(df)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=123)
    
    # Train models
    models = {
        'Lasso': lasso_training(X_train, y_train),
        'Ridge': ridge_training(X_train, y_train),
        'Linear Regression': linear_regression_training(X_train, y_train)[0],  # Get model only
        'ElasticNet': elastic_net_training(X_train, y_train),
        'Random Forest': random_forest_training(X_train, y_train),
        'SVR': svr_training(X_train, y_train),
        'Gradient Boosting': gradient_boost_training(X_train, y_train),
        'XGBoost': xgboost_training(X_train, y_train)
    }

    # Evaluate all models and store their metrics
    metrics = {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}
    
    # Select and print the best model
    best_model, best_metrics = select_best_model(metrics)
    print(f'The Best Model is: {best_model}')
    print(f"The Best Metrics are :")
    print(f"  - MAE: {best_metrics['MAE']:,.2f}")
    print(f"  - MSE: {best_metrics['MSE']:,.2f}")
    print(f"  - R2 : {best_metrics['R2']:.4f}")
