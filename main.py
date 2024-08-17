import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv('./data/USA_Housing.csv')

# Visialize data
sns.pairplot(df)

sns.distplot(df['Price'])

X = df.drop(['Price', 'Address'], axis=1)

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

lr = LinearRegression()
lr.fit(X_train, y_train)

cdf = pd.DataFrame(lr.coef_, X.columns, columns=['Coeff'])
print(cdf)


