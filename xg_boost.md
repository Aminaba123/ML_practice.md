
https://www.kaggle.com/code/rafiko1/enefit-xgboost-starter

```py

from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.reshape(pd.read_excel('COT_JPY.xlsx').values, (-1))
data = np.diff(data)

def data_preprocessing(data, num_lags, train_test_split):
    # Prepare the data for training
    x = []
    y = []
    for i in range(len(data) - num_lags):
        x.append(data[i:i + num_lags])
        y.append(data[i+ num_lags])
    # Convert the data to numpy arrays
    x = np.array(x)
    y = np.array(y)
    # Split the data into training and testing sets
    split_index = int(train_test_split * len(x))
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    
    return x_train, y_train, x_test, y_test 

x_train, y_train, x_test, y_test = data_preprocessing(data, 80, 0.80)

# Create the model
model = XGBRegressor(random_state = 0, n_estimators = 64, max_depth = 64)

# Fit the model to the data
model.fit(x_train, y_train)
y_pred_xgb = model.predict(x_test)

# Plotting
plt.plot(y_pred_xgb[-100:], label='Predicted Data | XGBoost', linestyle='--', marker = '.', color = 'orange')
plt.plot(y_test[-100:], label='True Data', marker = '.', alpha = 0.7, color = 'blue')
plt.legend()
plt.grid()
plt.axhline(y = 0, color = 'black', linestyle = '--')

same_sign_count = np.sum(np.sign(y_pred_xgb) == np.sign(y_test)) / len(y_test) * 100
print('Hit Ratio XGBoost = ', same_sign_count, '%')

```
