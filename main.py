import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the housing data into DataFrame
df = pd.read_csv('./data/USA_Housing.csv')

# Feature Engineering: Adding Room-to-Bedroom Ratio as a new feature
df['Room2Bedroom_ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']

# Define the feature variables and target variable
X = df.drop(['Price', 'Address'], axis=1, errors='ignore')
y = df['Price']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial Features to capture non-linear relationships
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=123)

# Initialize and train the Lasso regression model (L1 regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Print Lasso coefficients
lasso_coefficients = pd.DataFrame({
    'Feature': poly.get_feature_names_out(input_features=X.columns),
    'Coefficient': lasso.coef_
})
print('Lasso Coefficients:')
print(lasso_coefficients)

# Initialize and train the Linear Regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Cross-Validation to evaluate the Linear Regression model performance
cv_scores = cross_val_score(linear_regression, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
mean_cv_mae = -np.mean(cv_scores)
print(f'Average Cross-Validation MAE: {mean_cv_mae}')

# Create a DataFrame to view the coefficients of the Linear Regression model
lr_coefficients = pd.DataFrame({
    'Feature': poly.get_feature_names_out(input_features=X.columns),
    'Coefficient': linear_regression.coef_
})
print('Linear Regression Coefficients:')
print(lr_coefficients)

# Predict the target values using test data
y_pred = linear_regression.predict(X_test)

# Evaluate the model
evaluation_metrics = {
    'Mean Absolute Error (MAE)': mean_absolute_error(y_test, y_pred),
    'Mean Squared Error (MSE)': mean_squared_error(y_test, y_pred),
    'R-Squared': r2_score(y_test, y_pred)
}

print('Evaluation Metrics:')
for metric, value in evaluation_metrics.items():
    print(f'{metric}: {value}')
