# -*- coding: utf-8 -*-
"""Copy of Copy of Copy of Copy of Copy of DUntitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sSgOx3V7FCKpmgHbUKHNa9GPh3OVmsG9
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

pip install pandas

dataset= pd.read_csv('/content/Redvelvet - Redvelvet (2).csv')

dataset

data = {
    'Month': ['01/05/2021', '01/06/2021', '01/07/2021', '01/08/2021', '01/09/2021', '01/10/2021', '01/11/2021', '01/12/2021', '01/01/2022', '01/02/2022', '01/03/2022', '01/04/2022', '01/05/2022', '01/06/2022', '01/07/2022', '01/08/2022', '01/09/2022', '01/10/2022', '01/11/2022', '01/12/2022', '01/01/2023', '01/02/2023', '01/03/2023', '01/04/2023', '01/05/2023'],
   'Monthly Red Velvet Prediction': [25, 40, 28, 26, 35, 40, 55, 49, 60, 39, 48, 30, 42, 59, 44, 34, 37, 52, 62, 68, 69, 46, 58, 32, 50]
}

data_df = pd.DataFrame(data)

data_df['Month'] = pd.to_datetime(data_df['Month'], format='%d/%m/%Y')

data_df.set_index('Month', inplace=True)

data_df.describe()

plt.figure(figsize=(10, 6))
plt.plot(data_df.index, data_df['Monthly Red Velvet Prediction'])
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Original Time Series')
plt.show()

"""Cek Data Statisoner"""

def test_stationarity(timeseries):
    # Perform ADF test
    result = adfuller(timeseries, autolag='AIC')

    # Extract results
    adf_test_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    # Print results
    print("ADF Test Statistic:", adf_test_statistic)
    print("p-value:", p_value)
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"  {key}: {value}")

    # Check for stationarity based on p-value
    if p_value <= 0.05:
        print("Data is stationary (reject the null hypothesis)")
    else:
        print("Data is non-stationary (fail to reject the null hypothesis)")

test_stationarity(data_df['Monthly Red Velvet Prediction'])

def plot_acf_pacf(timeseries):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(timeseries, lags=10, ax=ax[0])
    plot_pacf(timeseries, lags=10, ax=ax[1])
    plt.tight_layout()
    plt.show()

test_stationarity(data_df['Monthly Red Velvet Prediction'])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf_pacf(data_df)

"""Split Dat"""

pip install pmdarima

train_data = data_df.loc['01/05/2021':'01/01/2023']
test_data = data_df.loc['01/02/2023':]

train_data

test_data

from pmdarima.arima import auto_arima

stepwise_model = auto_arima(data_df['Monthly Red Velvet Prediction'], seasonal=False, trace=True, start_p=1, start_q=1, max_p=9, max_q=9)

p = stepwise_model.order[0]
d = stepwise_model.order[1]
q = stepwise_model.order[2]
model = ARIMA(train_data, order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())

p = 4
d = 0
q = 2
model = ARIMA(train_data, order=(p, d, q))
model_fit = model.fit()

start_index = len(train_data)
end_index = len(train_data) + len(test_data) - 1
predictions = model_fit.predict(start=start_index, end=end_index, typ='levels')

model = ARIMA(data_df, order=(p, d, q))
model_fit = model.fit()

predictions

mse = mean_squared_error(test_data, predictions)
rmse = mean_squared_error(test_data, predictions, squared=False)
mae = mean_absolute_error(test_data, predictions)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)

predictions

print(model_fit.summary())

# Plot predictions vs actual values side-by-side
plt.figure(figsize=(16, 6))

# Plot Actual Data
plt.subplot(1, 2, 1)
plt.plot(data_df.index, data_df['Monthly Red Velvet Prediction'], color='red', label='Actual')
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('Actual Data')
plt.legend()

# Plot ARIMA Predictions
plt.subplot(1, 2, 2)
plt.plot(predictions.index, predictions, color='blue', label='Predictions')
plt.xlabel('Month')
plt.ylabel('Value')
plt.title('ARIMA Predictions')
plt.legend()

plt.tight_layout()
plt.show()

model_fit.plot_diagnostics(figsize=(10, 8))
plt.show()

import pandas as pd
data_df['Predictions'] = pd.Series(model_fit.fittedvalues, index=train_data.index).append(predictions)

plt.figure(figsize=(12, 6))
plt.plot(data_df.index, data_df['Monthly Red Velvet Prediction'], label='Actual', color='blue')
plt.plot(data_df.index, data_df['Predictions'], label='Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('ARIMA Predictions vs Actual')
plt.legend()
plt.show()

import pickle