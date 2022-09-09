import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('real-estate.csv')

df_test = df.sample(frac=0.20, random_state=60)
df_train = df.drop(df_test.index)

y = df['Y house price of unit area']
x = df[['X4 number of convenience stores', 'X5 latitude','X6 longitude']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.intercept_)

print(regressor.coef_)

feature_names = x.columns
model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(data = model_coefficients, index = feature_names, columns = ['Coefficient value'])
print(coefficients_df)

y_pred = regressor.predict(X_test)

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')