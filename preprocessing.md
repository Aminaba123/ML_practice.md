`Normalization`: also known as Min-Max scaling, transforms the data linearly to a specified range, typically between 0 and 1. It accomplishes this by subtracting the feature's minimum value and then dividing by the range (max - min). Mathematically, for a feature vector X, normalization transforms it as follows.

`Standardization`: on the other hand, rescales the data to have a mean of 0 and a standard deviation of 1. It accomplishes this by subtracting the feature's mean and then dividing by the standard deviation. Mathematically, for a feature vector X, standardization transforms it as follows

`Standardization VS NOrmalization`:

The primary distinction between normalization and standardization lies in their sensitivity to outliers. Normalization is susceptible to outliers because it scales the data based on the minimum and maximum values, which can be significantly affected by outliers. Standardization, on the other hand, is less sensitive to outliers because it scales the data based on the mean and standard deviation, which are more resilient to outliers.

Applications

Normalization is frequently used when the data has a well-defined range or when preserving relative distances between data points is critical. It is commonly used in image processing and neural networks.

Standardization is preferred when the data distribution is approximately Gaussian or when the presence of outliers is a concern. It is widely used in statistical analysis and machine learning algorithms that assume normally distributed data.

Since we have a high dimensionality data, scaling the data will help us with the Principal Component Analysis which we will take a look at next.
