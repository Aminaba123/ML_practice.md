
A Gentle Introduction to XGBoost Loss Functions

https://machinelearningmastery.com/xgboost-loss-functions/

Extreme Gradient Boosting (XGBoost) Ensemble in Python

https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/

Gradient Boosting with Scikit-Learn, XGBoost, LightGBM, and CatBoost

https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/

How to Use XGBoost for Time Series Forecasting

https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

Bagging vs Boosting

https://www.kaggle.com/code/prashant111/bagging-vs-boosting

Gradient Boosting and XGBoost

https://medium.com/@gabrieltseng/gradient-boosting-and-xgboost-c306c1bcfaf5

Home Credit Default Risk: XGBoost Model

https://www.kaggle.com/code/mahmoud86/home-credit-default-risk-xgboost-model

XGBoost + k-fold CV + Feature Importance

https://www.kaggle.com/code/prashant111/xgboost-k-fold-cv-feature-importance

A Guide on XGBoost hyperparameters tuning

https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning

Calibrated Xgboost - powershap my feats

https://www.kaggle.com/code/slythe/calibrated-xgboost-powershap-my-feats

Using XGBoost to predict probability

https://www.ikigailabs.io/multivariate-time-series-forecasting-in-python-settings/xgboost-predict-probability

Hyperparameter tuning in XGBoost

https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f

Does XGBoost handle multicollinearity by itself?

https://datascience.stackexchange.com/questions/12554/does-xgboost-handle-multicollinearity-by-itself

Calibrated Xgboost w/ +2000 features (tsflex)

https://www.kaggle.com/code/slythe/calibrated-xgboost-w-2000-features-tsflex

Optimization of XGBoost

https://www.kaggle.com/code/dstuerzer/optimization-of-xgboost

xgb, catboost & Isotonic Regression

https://www.kaggle.com/code/magnussesodia/ps-s3e10-xgb-catboost-isotonic-regression



https://www.kaggle.com/code/rafiko1/enefit-xgboost-starter

https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html

https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py

https://stackoverflow.com/questions/56904840/xgboost-and-cross-validation-in-parallel

https://stackoverflow.com/questions/57986259/multiclass-classification-with-xgboost-classifier

https://stackoverflow.com/questions/67868420/xgboost-for-multiclassification-and-imbalanced-data?rq=3

https://stackoverflow.com/questions/40968348/xgboost-dealing-with-imbalanced-classification-data?rq=3

https://stackoverflow.com/questions/57986259/multiclass-classification-with-xgboost-classifier?rq=3

https://stackoverflow.com/questions/61082381/xgboost-produce-prediction-result-and-probability?noredirect=1&lq=1

How to stack models ?

https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/

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
