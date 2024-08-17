import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the housing data into a DataFrame
df = pd.read_csv('./data/USA_Housing.csv')

# Visualize the relationship between features with pair plots
sns.pairplot(df)

# Visualize the distribution of the target variable
sns.histplot(df['Price'])

# Define the feature variables (X) and target variable (y)
# Dropping 'Price' as its tje target variable, and 'Address' as the it's irrelevant variable
X = df.drop(['Price', 'Address'], axis=1)
y = df['Price']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.2, random_state=123)

# Initialize the Linear Regression Model
lr = LinearRegression()

# Train the model using the training data
lr.fit(X_train, y_train)

# Create a DataFrame to view hthe coefficients of the model
cdf = pd.DataFrame(lr.coef_, X.columns, columns=['Coeff'])
print(cdf)

# Predict the target values (House prices) using the test data

y_pred = lr.predict(X_test)

# Plot the actual vs predicted values to vizualize model performance
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')

# Show plots
plt.show

# Evaluate the model using mean absolute error(MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Evaluate the mdoel using mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error: {mse}')
