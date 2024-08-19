import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRFRegressor
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

def handle_outliers(df, column_name, lower_quantile=0.01, upper_quantile=0.99):
    """Handle outliers by capping them at the specified quantiles."""
    lower_bound = df[column_name].quantile(lower_quantile)
    upper_bound = df[column_name].quantile(upper_quantile)
    df[column_name] = np.clip(df[column_name], lower_bound, upper_bound)
    logging.info(f'Outliers in {column_name} handled by capping at {lower_quantile} and {upper_quantile} quantiles.')
    return df

def feature_engineering(df, handle_outliers=False):
    """Perform feature engineering on the dataset, with optional outlier handling."""
    df['Room2Bedroom_ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']
    
    if handle_outliers:
        df = handle_outliers(df, 'Price')
    
    # Filter out non-numeric columns before skewness correction
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Handle skewness in numeric features
    skewness_threshold = 0.75
    skewed_feats = numeric_df.apply(lambda x: x.skew()).sort_values(ascending=False)
    skewed_features = skewed_feats[abs(skewed_feats) > skewness_threshold].index
    
    if len(skewed_features) > 0:
        pt = PowerTransformer(method='yeo-johnson')
        df[skewed_features] = pt.fit_transform(df[skewed_features])
            
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
    
def select_best_model(eval_metrics):
    """Select the best model based on MAE."""
    best_model = min(eval_metrics, key=lambda k: eval_metrics[k]['MAE'])
    best_metrics = eval_metrics[best_model]
    return best_model, best_metrics

if __name__ == "__main__":
    file_path = './data/USA_Housing.csv'
    
    # Load and process the data
    df = load_data(file_path)
    
    # Set to True if you want to handle outliers
    handle_outliers_flag = True
    
    df = feature_engineering(df, handle_outliers=handle_outliers_flag)
    
    # Preprocess the data with PCA applied
    y, X_processed = preprocess_data(df)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=123)
    
    # Define the models
    models = {
        'Lasso': Lasso(alpha=0.1, random_state=12),
        'Ridge': Ridge(alpha=0.1, random_state=12),
        'Linear Regression': LinearRegression(),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=12),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=12),
        'SVR': SVR(kernel='rbf', C=1.0),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=123),
        'XGBoost': XGBRFRegressor(n_estimators=100, learning_rate=0.1, random_state=12)
    }

    # Train and evaluate models
    metrics = {}
    for name, model in models.items():
        trained_model, _ = train_model(model, X_train, y_train)
        metrics[name] = evaluate_model(trained_model, X_test, y_test)
    
    # Select and print the best model
    best_model, best_metrics = select_best_model(metrics)
    print(f'The Best Model is: {best_model}')
    print(f"The Best Metrics are :")
    print(f"  - MAE: {best_metrics['MAE']:,.2f}")
    print(f"  - MSE: {best_metrics['MSE']:,.2f}")
    print(f"  - R2 : {best_metrics['R2']*100:.2f}%")
