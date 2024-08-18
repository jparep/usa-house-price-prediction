import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the housing data into DataFrame
df = pd.read_csv('./data/USA_Housing.csv')

# Feature Engineering: Adding Room-to-Bedroom Ration as a new features
df['Room2Bedroom_ratio'] = df['Avg. Area Number of Rooms'] / df['Avg. Area Number of Bedrooms']

# Define the feature variables and target variable
X = df.drop(['Price', 'Address'])
y = df['Price']

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ploynomial Features to capture non-linear relationships
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

#  Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=123)

# Initalize Model
lr = LinearRegression()

# Train the model using training data
lr.fit(X_train, y_train)

# Cross-Validation to evaluate the mdoel performance
cv_score = cross_val_predict(lr, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f'Cross-Validation MAE Scores: {-cv_score}')
print(f'Average Cross-Validation MAE: {-np.mean(cv_score)}')

# Create a DataFrame to view the coeffient of the model
cdf = pd.DataFrame(lr.coef_, poly.get_feature_names_out(input_features=X.columns), columns='Coefficients')

# Predict the target values using test data
y_pred = lr.predict(X_test)

# Evaluate model using Meam Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Evaluate model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Evalaute Model using R-Squared
r2 = r2_score(y_test, y_pred)
print(f'R-Squared: {r2}')
