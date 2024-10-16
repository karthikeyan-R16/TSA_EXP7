# Ex.No: 07                                       AUTO REGRESSIVE MODEL
# DEVELOPED BY: Karthikeyan R
# REGISTER NO: 212222240045
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:

# Import necessary libraries
```
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```
# Load the dataset
```
file_path = '/content/rainfall.csv'
df = pd.read_csv(file_path)
```
# Convert 'date' column to datetime format
```
df['date'] = pd.to_datetime(df['date'])
```
# Set the date as the index (assuming we want to predict 'total_sales')
```
df.set_index('date', inplace=True)
```

# Select the target variable (for example, 'total_sales') for the AR model
```
default_column_name = df.columns[0]  # Assuming rainfall data is in the first column
series = df[default_column_name]
print(f"Column 'total_rainfall' not found. Using '{default_column_name}' instead.")
```
# Perform Augmented Dickey-Fuller (ADF) test to check stationarity
```
adf_result = adfuller(series)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
```

# Split the data into training and testing sets (80% training, 20% testing)
```
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]
```
# Plot ACF and PACF
```
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(series, lags=30, ax=plt.gca())
plt.subplot(122)
plot_pacf(series, lags=30, ax=plt.gca())
plt.show()
```
# Fit AutoRegressive (AR) model with 13 lags
```
model = AutoReg(train, lags=13)
model_fitted = model.fit()
```
# Make predictions on the test set
```
predictions = model_fitted.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
```
# Compare predictions with actual test data
```
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.legend()
plt.title('AR Model - Actual vs Predicted')
plt.show()
```

# Calculate Mean Squared Error (MSE)
```
mse = mean_squared_error(test, predictions)
print(f"Mean Squared Error: {mse}")
```
# OUTPUT:

 # PACF - ACF
![image](https://github.com/user-attachments/assets/a5c76cbc-023b-4515-8422-1dc794d588ef)



# PREDICTION
![image](https://github.com/user-attachments/assets/c18464b0-8da2-4549-b5be-2ac36c84d68b)



# FINIAL PREDICTION
![image](https://github.com/user-attachments/assets/77a98a58-6345-4c34-b8ad-37d92e85e005)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
