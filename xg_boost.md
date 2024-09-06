
https://www.kaggle.com/code/rafiko1/enefit-xgboost-starter

https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html

https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py

https://stackoverflow.com/questions/56904840/xgboost-and-cross-validation-in-parallel

https://stackoverflow.com/questions/57986259/multiclass-classification-with-xgboost-classifier

https://stackoverflow.com/questions/67868420/xgboost-for-multiclassification-and-imbalanced-data?rq=3

https://stackoverflow.com/questions/40968348/xgboost-dealing-with-imbalanced-classification-data?rq=3

https://stackoverflow.com/questions/57986259/multiclass-classification-with-xgboost-classifier?rq=3

https://stackoverflow.com/questions/61082381/xgboost-produce-prediction-result-and-probability?noredirect=1&lq=1

#
Tune Learning Rate for Gradient Boosting with XGBoost

https://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/

#

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
