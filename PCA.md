`Why do we need dimensionality reduction?`

Dimensionality reduction is a crucial step in the data preprocessing pipeline for machine learning, particularly when dealing with high-dimensional data. By reducing the number of features while retaining essential information, we can improve model performance, enhance visualization, decrease computational costs, and increase interpretability. The choice of dimensionality reduction technique depends on the specific characteristics of the dataset and the goals of the analysis.

Whay in high-dimensional spaces, data can become sparse?

In high-dimensional spaces, data can become sparse due to several reasons related to the nature of the data and the curse of dimensionality.

`what is the curse of dimensionality ?`

- As the number of dimensions (features) increases, the volume of the space increases exponentially. This means that data points become more spread out.
- For example, in a 2D space, you can easily visualize points close together. However, as you add more dimensions, the number of potential positions for each data point increases drastically, making it less likely that any two points will be close together.

`why it leads to sparsity?`

- In high-dimensional space, the data points occupy a larger volume, but the actual number of data points remains the same or increases at a slower rate. As a result, the density of the data points in that space decreases.
- When there are fewer data points relative to the space they occupy, it creates "gaps" between points, leading to sparsity.

High-dimensional datasets often include many features that are irrelevant or redundant. These unnecessary features can dilute the meaningful structure in the data, making the data appear more sparse.

if you have a dataset with many features but only a few of them actually contribute to the outcome, the true signal may be buried in noise.

For example in clustering why we need to have diminsioality reduction ?

In high dimensions, the concept of distance changes. For example, in high-dimensional space, all points tend to become equidistant from each other. This phenomenon can make it difficult to find meaningful clusters or relationships.

As dimensions increase, the distance between points tends to increase, reducing the likelihood of finding neighbors, which is critical for algorithms that rely on neighborhood relationships (e.g., clustering, k-NN)

If the number of data points does not grow proportionately with the number of dimensions, the data will inevitably become sparse. This is particularly evident in fields like image processing, genomics, and other domains where high-dimensional data is common.

Model Performance: Sparsity can lead to overfitting in machine learning models, where the model learns the noise in the training data rather than the underlying patterns.
Computational Complexity: Sparse data can also increase the computational burden, as many algorithms (like clustering) may struggle to find meaningful patterns or relationships when data is dispersed.


`Which Dimensionality Reduction Method to Use?`

As we said :

Data can become sparse in high-dimensional spaces due to the curse of dimensionality, which refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces that do not occur in low-dimensional settings.

- Exponential Increase in Volume:
As the number of dimensions increases, the volume of the space increases exponentially. For example, while a point in a 1D space can be described with a single number, a point in a 10D space requires 10 numbers. As dimensions increase, the space contains far more possible points than can be covered by the data.

- Distance Metrics:
In high-dimensional spaces, the distance between data points becomes less meaningful. The distance between points tends to converge, making it hard to distinguish between them. For instance, the difference between the nearest and farthest neighbor can become negligible, leading to difficulties in clustering and classification.

- Increased Data Requirement:

To maintain the same density of points in a high-dimensional space as in a lower-dimensional space, an exponentially larger number of data points is required. In many cases, we do not have enough data to adequately fill the space, leading to sparse datasets.


The choice of dimensionality reduction method depends on several factors, including the nature of the data, the specific use case, and the analysis objectives.

- Principal Component Analysis (PCA):

- - Use When: You have linear relationships in your data and want to reduce dimensions while retaining variance.
- - Benefits: PCA is effective for data compression, visualization, and feature extraction.
- - Limitations: PCA assumes linearity and can be sensitive to outliers.
 
- t-Distributed Stochastic Neighbor Embedding (t-SNE):

- - Use When: You want to visualize high-dimensional data in 2D or 3D while preserving local structures.
- - Benefits: Excellent for visualizing clusters in the data.
- - Limitations: Computationally intensive and not suitable for large datasets. Not ideal for preserving global structure.

- Linear Discriminant Analysis (LDA):

- - Use When: You have labeled data and want to maximize class separability while reducing dimensions.
- - Benefits: Effective for supervised dimensionality reduction.
- - Limitations: Requires labeled data and assumes normal distribution of features.

- Autoencoders:

- - Use When: You have a large dataset and want to learn non-linear representations.
- - Benefits: Flexible and can capture complex relationships in the data.
- - Limitations: Requires more computational resources and tuning.

- UMAP (Uniform Manifold Approximation and Projection):

- - Use When: You want a balance between preserving local and global structure in data visualization.
- - Benefits: Efficient and effective for large datasets, often outperforming t-SNE in terms of speed and quality of embedding.
- - Limitations: Requires some parameter tuning.

#

`Using PCA`

We use dimensionality reduction techniques to simplify our data while retaining the essential information. Among various methods such as KernelPCA, ICA, ISOMAP, TSNE, and UMAP

- PCA is an excellent starting point because it works well in capturing linear relationships in the data, which is particularly relevant given the multicollinearity we identified in our dataset. It allows us to reduce the number of features in our dataset while still retaining a significant amount of the information, thus making our clustering analysis potentially more accurate and interpretable. Moreover, it is computationally efficient, which means it won't significantly increase the processing time.

- After applying PCA, if we find that the first few components do not capture a significant amount of variance, indicating a loss of vital information, we might consider exploring other non-linear methods. These methods can potentially provide a more nuanced approach to dimensionality reduction, capturing complex patterns that PCA might miss, albeit at the cost of increased computational time and complexity.

I will apply PCA on all the available components and plot the cumulative variance explained by them. This process will allow me to visualize how much variance each additional principal component can explain, thereby helping me to pinpoint the optimal number of components to retain for the analysis.

- The plot and the cumulative explained variance values indicate how much of the total variance in the dataset is captured by each principal component, as well as the cumulative variance explained by the first n components.

- The first component explains approximately 28% of the variance.

- The first two components together explain about 49% of the variance.

- The first three components explain approximately 61% of the variance, and so on.

- To choose the optimal number of components, we generally look for a point where adding another component doesn't significantly increase the cumulative explained variance, often referred to as the "elbow point" in the curve.

- From the plot, we can see that the increase in cumulative variance starts to slow down after the 6th component (which captures about 81% of the total variance).

- Considering the context of customer segmentation, we want to retain a sufficient amount of information to identify distinct customer groups effectively. Therefore, retaining the first 6 components might be a balanced choice, as they together explain a substantial portion of the total variance while reducing the dimensionality of the dataset.

- We can drop these columns from our data and then we can use PCA to find which column explain the most variance in our data. By doing this, we will be helping the PCA algorithm computationally.

#
https://www.kaggle.com/code/bhatnagardaksh/pca-and-lda-implementation

Principal components are linear combinations of the original variables in the dataset. They are new features that capture the most significant patterns in the data.

PCA does two main thing :

`Dimensionality Reduction`: PCA reduces the number of variables, simplifying data visualization and analysis. This makes it easier to identify patterns and relationships within the data.

`Overfitting Prevention`: By eliminating highly correlated features, PCA reduces the risk of overfitting in machine learning models. Overfitting occurs when a model becomes too closely aligned with the training data, leading to poor performance on new data

What are the priciple componets?

Principal components are linear combinations of the original variables in the dataset. They are new features that capture the most significant patterns in the data.


`PCA achieves these goals by **maximizing the variance** of the projected data on a given axis while **minimizing the reconstruction error** or residuals. It identifies principal components, directions that capture the most significant variations in the data.` Projecting data onto these principal components maximizes variance and minimizes reconstruction error, effectively reducing dimensionality without losing crucial information.

The below image aptly illustrates PCA's objective. The red dots represent data points, and the arrows represent principal components. Longer arrows correspond to directions with higher variance, while shorter arrows correspond to directions with lower variance. PCA projects data onto these principal components, reducing dimensionality while preserving essential information.


`Creating Covariance Matrix and finding Eigenvalues and Eigen Vectors`

An eigenvector (eigen is German for "typical"; we could translate eigenvector to "characteristic vector") is a special vector  
such that when it is transformed by some matrix (let's say  
A), the product has the exact same direction as  v. An eigenvalue is a scalar (traditionally represented as  λ ) that simply scales the eigenvector v such that the following equation is satisfied:

```txt
A = square matrix (equal rows & columns)

→
v = non-zero vector

λ = scalar value or eigenvalue

```

An eigenvector of a square matrix  A is a non-zero vector v that, when multiplied by the matrix  A , results in a scalar multiple of itself. Basically, Eigenvectors are vectors that, when transformed by the matrix  A , only change in magnitude, not in direction.


For example : consider a 2D eigenvector [1, 1] and its corresponding matrix A. If A scales the vector by a factor of 2, the transformed vector will be [2, 2]. The vector has doubled in length, but its direction remains the same – it still points along a 45-degree angle from the  x-axis.

`Calculation of Eigenvalues and Eigenvectors`

Let's see how we calculate the eigenvalues and the eigenvectors. To find the eigenvectors of the matrix A , we first need to find the eigenvalues. The eigenvalues are the values  λ that satisfy the equation:

- The principal components are derived from the eigenvectors of the covariance matrix of the original dataset.
- Each eigenvector corresponds to a direction in the feature space, while the eigenvalue associated with each eigenvector indicates the amount of variance captured by that component. The larger the eigenvalue, the more variance is captured by the corresponding principal component.

- Mathematically, if 𝜆1 is the largest eigenvalue, then PC1 corresponds to the eigenvector associated with 𝜆1. PC2 corresponds to the eigenvector associated with the second-largest eigenvalue 𝜆2 , and so forth.

`Properties of Principal Components`

`Orthogonality`: The principal components are orthogonal (perpendicular) to each other. This means they are uncorrelated and provide unique information about the data. Each component captures a different aspect of variance in the dataset.

`Dimensionality Reduction`: By selecting only the first few principal components (those with the highest variance), we can effectively reduce the dimensionality of the dataset while retaining most of the original information.

`Variance Explained`: The proportion of variance explained by each principal component can be calculated as

This allows us to understand how many components are needed to capture a significant portion of the total variance.

`Why do we calculate the covariance matrix`

The covariance matrix is used to calculate eigenvectors and eigenvalues in principal component analysis (PCA) because it captures the variances and covariances between the features in a dataset. PCA aims to identify the principal components, which are the directions of maximum variance in the data. These directions correspond to the eigenvectors of the covariance matrix, and the corresponding eigenvalues represent the magnitudes of variance along those directions.

The covariance matrix is a square matrix where each element represents the covariance between two features. The diagonal elements represent the variances of individual features, while the off-diagonal elements represent the covariances between different features.

By calculating the eigenvectors and eigenvalues of the covariance matrix, PCA identifies the directions of maximum variance in the data. These directions are the principal components, and they represent the most significant patterns of variation in the dataset. Projecting the data onto these principal components reduces dimensionality while preserving as much of the original information as possible.

`Displaying the Linear Transformations`

Let's now transform our original data now that we have found out our principal components. The operation of transforming the original data using principal components is called principal component projection. It involves projecting the original data onto the principal components, which are the directions of maximum variance in the data. This projection effectively reduces the dimensionality of the data while preserving as much of the original information as possible.

In this transformation, what we do is do a dot-product between the original data and the Principal components we just found out.

The process involves multiplying the centered data matrix by the matrix of selected eigenvectors, resulting in a transformed data matrix with lower dimensionality. This transformed data captures the most significant variations in the original data while reducing the number of variables, making it easier to analyze and visualize.


We have chosen 5 components since they explain the 90% of the explained variance in our data. The transpose is being done to enable the matrix multiplication. The shape of the dataframe and the Principal Components array is different and to be able to multiply the matrices, the length of the rows of one matrix should be equal to the length of the columns of another matrix



```py

import math

# Function to calculate the mean of a list
def mean(data):
    return sum(data) / len(data)

# Function to center the data
def center_data(data):
    means = [mean(col) for col in zip(*data)]
    centered = [[data[i][j] - means[j] for j in range(len(data[0]))] for i in range(len(data))]
    return centered

# Function to calculate the covariance matrix
def covariance_matrix(data):
    n = len(data)
    centered = center_data(data)
    cov_matrix = [[0] * len(data[0]) for _ in range(len(data[0]))]
    
    for i in range(len(data[0])):
        for j in range(len(data[0])):
            cov_matrix[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / (n - 1)
    
    return cov_matrix

# Function to calculate the eigenvalues and eigenvectors
def eigen_decomposition(cov_matrix):
    # Placeholder for eigenvalues and eigenvectors
    eigenvalues = []
    eigenvectors = []
    
    # For simplicity, we will use numpy for eigenvalue/eigenvector calculation
    import numpy as np
    
    # Convert covariance matrix to numpy array for eigen decomposition
    cov_matrix_np = np.array(cov_matrix)
    vals, vecs = np.linalg.eig(cov_matrix_np)
    
    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(vals)[::-1]  # Sort in descending order
    for index in sorted_indices:
        eigenvalues.append(vals[index])
        eigenvectors.append(vecs[:, index].tolist())  # Convert to list for consistency

    return eigenvalues, eigenvectors

# Function to project the data onto the selected principal components
def transform_data(data, eigenvectors, k):
    return [[sum(data_point[j] * eigenvectors[j][i] for j in range(len(data_point))) for i in range(k)] for data_point in data]

# Main PCA function
def PCA(data, k):
    # Step 1: Center the data
    centered_data = center_data(data)

    # Step 2: Calculate the covariance matrix
    cov_matrix = covariance_matrix(centered_data)

    # Step 3: Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)

    # Step 4: Project the data onto the top k eigenvectors
    transformed_data = transform_data(centered_data, eigenvectors, k)

    return transformed_data, eigenvalues, eigenvectors

# Example usage with a 3x3 data matrix
data = [
    [2.5, 2.4, 3.1],
    [0.5, 0.7, 0.9],
    [2.2, 2.9, 3.3]
]

# Perform PCA to reduce to 2 dimensions
k = 2
transformed_data, eigenvalues, eigenvectors = PCA(data, k)

# Output the results
print("Transformed Data (PCA):")
for row in transformed_data:
    print(row)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
for vec in eigenvectors:
    print(vec)

```

