Simlified Clustering with code and example

https://www.kaggle.com/code/daniilkrasnoproshin/k-means-simplified-expedition-into-clustering


Customer Segmentation & Recommendation System

https://www.kaggle.com/code/farzadnekouei/customer-segmentation-recommendation-system#Step-3.7-%7C-Outlier-Treatment

10 clustering algo in python 

https://machinelearningmastery.com/clustering-algorithms-with-python/


### K-Means 

Clustering is an unsupervised learning technique that involves grouping a set of objects (data points) into clusters based on similarity. The goal is to ensure that objects within the same cluster are more similar to each other than to those in other clusters.

`Common Clustering Techniques`

K-means Clustering:

- A partitioning method that divides data into 𝑘 clusters, where each data point belongs to the cluster with the nearest mean (centroid).
- Iteratively updates centroids and reassigns points until convergence.

Hierarchical Clustering:

- Creates a hierarchy of clusters using either agglomerative (bottom-up) or divisive (top-down) approaches.
- Produces a dendrogram to visualize the clustering structure.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise):

- Groups together points that are closely packed while marking points in low-density regions as outliers.
- Effective for discovering clusters of arbitrary shape and dealing with noise.

Mean Shift

- A non-parametric clustering technique that seeks modes (high-density regions) in the data.
- Moves points towards the highest density of data points iteratively.

Gaussian Mixture Models (GMM):

- A probabilistic model that assumes that the data is generated from a mixture of several Gaussian distributions.
- Useful for soft clustering, where points can belong to multiple clusters with varying probabilities.

Finding the Optimal k:

Finding the optimal number of clusters k :

in K-means clustering is essential for meaningful segmentation. Here are common methods:

Elbow Method:

- Fit the K-means algorithm for a range of k values (e.g., from 1 to 10).
- Plot the Within-Cluster Sum of Squares (WCSS) against different k values.
- Identify the "elbow point" in the plot where the rate of decrease in WCSS slows down significantly. This point suggests a suitable k.

Silhouette Method:

- Calculate the Silhouette Score for various k values.
- The Silhouette Score measures how similar a data point is to its own cluster compared to other clusters.
- Choose the k with the highest average Silhouette Score.

Davies-Bouldin Index:

- Evaluate the average similarity of each cluster with its most similar cluster.
- A lower Davies-Bouldin Index indicates better clustering.

Gap Statistic:

- Compare the clustering result on the observed data with that on a reference dataset (null distribution).
- The optimal k is where the gap between the two is maximized.

Silhouette Method and Its Relation to K-means:

- Silhouette Score: The Silhouette Score for a single data point is calculated as:
- The overall Silhouette Score is the average of the Silhouette Scores for all data points.

Relation to K-means:

- After performing K-means clustering, you can calculate the Silhouette Score to evaluate the quality of the clusters.
- A higher Silhouette Score indicates that clusters are well-separated, while a score close to zero suggests overlapping clusters.

Evaluating Clustering

- Evaluating clustering involves assessing how well the clusters represent the underlying data structure. Common evaluation metrics include:

- Silhouette Score: As described earlier, it measures the quality of clustering. Higher scores indicate better-defined clusters.

Davies-Bouldin Index: Measures cluster separation and compactness. Lower values indicate better clustering.

Adjusted Rand Index (ARI): Compares the clustering results with a ground truth classification (if available) while adjusting for chance. Values range from -1 to 1, with higher values indicating better agreement.

Important Considerations in Clustering

Data Preprocessing: Normalize or standardize data to ensure that all features contribute equally to distance calculations.

Choosing the Right Algorithm: The choice of clustering algorithm can affect the results. For instance, K-means assumes spherical clusters, while DBSCAN can find arbitrary shapes.

Handling Noise and Outliers: Consider techniques that can handle noise and outliers, especially in real-world datasets.

Interpretability: Ensure that the clusters are interpretable and actionable for business purposes.

Scalability: Some algorithms scale better to large datasets than others. Consider the size of the dataset when choosing an algorithm.


#

https://www.kaggle.com/code/farzadnekouei/customer-segmentation-recommendation-system

In this project, we delve deep into the thriving sector of online retail by analyzing a transactional dataset from a UK-based retailer, available at the UCI Machine Learning Repository. This dataset documents all transactions between 2010 and 2011. Our primary objective is to amplify the efficiency of marketing strategies and boost sales through customer segmentation. We aim to transform the transactional data into a customer-centric dataset by creating new features that will facilitate the segmentation of customers into distinct groups using the K-means clustering algorithm. This segmentation will allow us to understand the distinct profiles and preferences of different customer groups. Building upon this, we intend to develop a recommendation system that will suggest top-selling products to customers within each segment who haven't purchased those items yet, ultimately enhancing marketing efficacy and fostering increased sales.

Is an unsupervised machine learning algorithm that clusters data into a specified number of groups (K) by minimizing the `within-cluster sum-of-squares (WCSS)`, also known as `inertia`. The algorithm iteratively assigns each data point to the nearest `centroid`, then updates the centroids by calculating the mean of all assigned points. The process repeats until convergence or a stopping criterion is reached.


`K-Means` is an unsupervised machine learning algorithm that clusters data into a specified number of groups (K) by `minimizing` the `within-cluster sum-of-squares (WCSS)`, also known as `inertia`. The algorithm iteratively assigns each data point to the nearest `centroid`, then updates the centroids by calculating the mean of all assigned points. The process repeats until convergence or a stopping criterion is reached.

#### Main drawbacks 

Inertia is influenced by the number of dimensions: The value of inertia tends to increase in high-dimensional spaces due to the curse of dimensionality, which can distort the Euclidean distances between data points.
Solution: Performing dimensionality reduction, such as PCA, before applying K-means to alleviate this issue and speed up computations.

Dependence on Initial Centroid Placement: The K-means algorithm might find a local minimum instead of a global minimum, based on where the centroids are initially placed.
Solution: To enhance the likelihood of locating the global minimum, we can employ the k-means++ initialization method

Requires specifying the number of clusters: K-means requires specifying the number of clusters (K) beforehand, which may not be known in advance.
Solution: Using methods such as the elbow method and silhouette analysis to estimate the optimal number of clusters.

Sensitivity to unevenly sized or sparse clusters: K-means might struggle with clusters of different sizes or densities.

Solution: Increasing the number of random initializations (n_init) or consider using algorithms that handle unevenly sized clusters better, like GMM or DBSCAN.

Assumes convex and isotropic clusters: K-means assumes that clusters are spherical and have similar variances, which is not always the case. It may struggle with elongated or irregularly shaped clusters.

Assumes convex and isotropic clusters: K-means assumes that clusters are spherical and have similar variances, which is not always the case. It may struggle with elongated or irregularly shaped clusters.

`Determining the Optimal Number of Clusters`

To ascertain the optimal number of clusters (k) for segmenting customers, I will explore two renowned methods:

- Elbow Method

- Silhouette Method

It's common to utilize both methods in practice to corroborate the results.

`What is the Elbow Method?`

The Elbow Method is a technique for identifying the ideal number of clusters in a dataset. It involves iterating through the data, generating clusters for various values of k. The k-means algorithm calculates the sum of squared distances between each data point and its assigned cluster centroid, known as the inertia or WCSS score. By plotting the inertia score against the k value, we create a graph that typically exhibits an elbow shape, hence the name "Elbow Method". The elbow point represents the k-value where the reduction in inertia achieved by increasing k becomes negligible, indicating the optimal stopping point for the number of clusters.

`Optimal k Value: Elbow Method Insights`

The optimal value of k for the KMeans clustering algorithm can be found at the elbow point. Using the YellowBrick library for the Elbow method, we observe that the suggested optimal k value is 5. However, we don't have a very distinct elbow point in this case, which is common in real-world data. From the plot, we can see that the inertia continues to decrease significantly up to k=5, indicating that the optimum value of k could be between 3 and 7. To choose the best k within this range, we can employ the silhouette analysis, another cluster quality evaluation method. Additionally, incorporating business insights can help determine a practical k value.

`What is the Silhouette Method?`

The Silhouette Method is an approach to find the optimal number of clusters in a dataset by evaluating the consistency within clusters and their separation from other clusters. It computes the silhouette coefficient for each data point, which measures how similar a point is to its own cluster compared to other clusters.

`What is the Silhouette Coefficient?`

To determine the silhouette coefficient for a given point i, follow these steps:

Calculate a(i): Compute the average distance between point i and all other points within its cluster.
Calculate b(i): Compute the average distance between point i and all points in the nearest cluster to its own.
Compute the silhouette coefficient, s(i), for point i using the following formula:

Note: The silhouette coefficient quantifies the similarity of a point to its own cluster (cohesion) relative to its separation from other clusters. This value ranges from -1 to 1, with higher values signifying that the point is well aligned with its cluster and has a low similarity to neighboring clusters.


`What is the Silhouette Score?`

The silhouette score is the average silhouette coefficient calculated for all data points in a dataset. It provides an overall assessment of the clustering quality, taking into account both cohesion within clusters and separation between clusters. A higher silhouette score indicates a better clustering configuration.

`What are the Advantages of Silhouette Method over the Elbow Method?`

- The Silhouette Method evaluates cluster quality by considering both the cohesion within clusters and their separation from other clusters. This provides a more comprehensive measure of clustering performance compared to the Elbow Method, which only considers the inertia (sum of squared distances within clusters).

- The Silhouette Method produces a silhouette score that directly quantifies the quality of clustering, making it easier to compare different values of k. In contrast, the Elbow Method relies on the subjective interpretation of the elbow point, which can be less reliable in cases where the plot does not show a clear elbow.

- The Silhouette Method generates a visual representation of silhouette coefficients for each data point, allowing for easier identification of fluctuations and outliers within clusters. This helps in determining the optimal number of clusters with higher confidence, as opposed to the Elbow Method, which relies on visual inspection of the inertia plot.

`Methodology`

In the following analysis:

I will initially choose a range of 2-6 for the number of clusters (k) based on the Elbow method from the previous section. Next, I will plot Silhouette scores for each k value to determine the one with the highest score.
Subsequently, to fine-tune the selection of the most appropriate k, I will generate Silhouette plots that visually display the silhouette coefficients for each data point within various clusters.

`Guidelines to Interpret Silhouette Plots and Determine the Optimal K`:

- Silhouette Score Width:

- - Wide Widths (closer to +1): Indicate that the data points in the cluster are well separated from points in other clusters, suggesting well-defined clusters.
- - Narrow Widths (closer to -1): Show that data points in the cluster are not distinctly separated from other clusters, indicating poorly defined clusters.

- Average Silhouette Score:

- - High Average Width: A cluster with a high average silhouette score indicates well-separated clusters.
- - Low Average Width: A cluster with a low average silhouette score indicates poor separation between clusters.

Uniformity in Cluster Size:

Cluster Thickness:

Uniform Thickness: Indicates that clusters have a roughly equal number of data points, suggesting a balanced clustering structure.

Variable Thickness: Signifies an imbalance in the data point distribution across clusters, with some clusters having many data points and others too few.

Peaks in Average Silhouette Score:

Clear Peaks: A clear peak in the average silhouette score plot for a specific ( k ) value indicates this ( k ) might be optimal.

Minimize Fluctuations in Silhouette Plot Widths:

Uniform Widths: Seek silhouette plots with similar widths across clusters, suggesting a more balanced and optimal clustering.
Variable Widths: Avoid wide fluctuations in silhouette plot widths, indicating that clusters are not well-defined and may vary in compactness.

Optimal Cluster Selection:

Maximize the Overall Average Silhouette Score: Choose the ( k ) value that gives the highest average silhouette score across all clusters, indicating well-defined clusters.

Avoid Below-Average Silhouette Scores: Ensure most clusters have above-average silhouette scores to prevent suboptimal clustering structures.

fter determining the optimal number of clusters (which is 3 in our case) using elbow and silhouette analyses, I move onto the evaluation step to assess the quality of the clusters formed. This step is essential to validate the effectiveness of the clustering and to ensure that the clusters are coherent and well-separated. The evaluation metrics and a visualization technique I plan to use are outlined below:

`Clustering Evaluation`

1️⃣ 3D Visualization of Top PCs : 

- 3D visualization. This will allow us to visually inspect the quality of separation and cohesion of clusters to some extent:

2️⃣ Cluster Distribution Visualization:  a bar plot to visualize the percentage of customers in each cluster, which helps in understanding if the clusters are balanced and significant:

3️⃣ Evaluation Metrics

- Silhouette Score
- Calinski Harabasz Score
- Davies Bouldin Score

`Inference` 

The distribution of customers across the clusters, as depicted by the bar plot, suggests a fairly balanced distribution with clusters 0 and 1 holding around 41% of customers each and cluster 2 accommodating approximately 18% of the customers.

This balanced distribution indicates that our clustering process has been largely successful in identifying meaningful patterns within the data, rather than merely grouping noise or outliers. It implies that each cluster represents a substantial and distinct segment of the customer base, thereby offering valuable insights for future business strategies.

Moreover, the fact that no cluster contains a very small percentage of customers, assures us that each cluster is significant and not just representing outliers or noise in the data. This setup allows for a more nuanced understanding and analysis of different customer segments, facilitating effective and informed decision-making.

`Evaluation Metrics`

To further scrutinize the quality of our clustering, I will employ the following metrics:

`Silhouette Score`: A measure to evaluate the separation distance between the clusters. Higher values indicate better cluster separation. It ranges from -1 to 1.
`Calinski Harabasz Score`: This score is used to evaluate the dispersion between and within clusters. A higher score indicates better defined clusters.
`Davies Bouldin Score`: It assesses the average similarity between each cluster and its most similar cluster. Lower values indicate better cluster separation.


Clustering Quality Inference

- The Silhouette Score of approximately 0.236, although not close to 1, still indicates a fair amount of separation between the clusters. It suggests that the clusters are somewhat distinct, but there might be slight overlaps between them. Generally, a score closer to 1 would be ideal, indicating more distinct and well-separated clusters.
- The Calinski Harabasz Score is 1257.17, which is considerably high, indicating that the clusters are well-defined. A higher score in this metric generally signals better cluster definitions, thus implying that our clustering has managed to find substantial structure in the data.
- The Davies Bouldin Score of 1.37 is a reasonable score, indicating a moderate level of similarity between each cluster and its most similar one. A lower score is generally better as it indicates less similarity between clusters, and thus, our score here suggests a decent separation between the clusters.

- Cluster Analysis and Profiling


# Clustering Evaluation Methods

This document summarizes the differences between two commonly used methods for determining the optimal number of clusters in clustering algorithms: the Elbow Method and the Silhouette Method.

## Summary of Differences

| Feature                  | Elbow Method                                 | Silhouette Method                             |
|--------------------------|----------------------------------------------|----------------------------------------------|
| **What It Measures**     | Within-Cluster Sum of Squares (WCSS)       | How well-separated the clusters are          |
| **Interpretation**       | Look for the "elbow" point in the plot     | Aim for the highest average Silhouette Score |
| **Visual Representation**| Plot of WCSS vs. \( k \)                    | Plot of average Silhouette Score vs. \( k \) |
| **Indication of Quality**| Diminishing returns in variance reduction    | Clarity and separation of clusters           |

## Description

### Elbow Method
- **Definition**: Identifies the optimal number of clusters by plotting WCSS against different values of \( k \).
- **Usage**: The "elbow" point indicates a suitable number of clusters where adding more clusters results in minimal variance reduction.

### Silhouette Method
- **Definition**: Evaluates the quality of clusters by measuring how similar each data point is to its own cluster compared to other clusters.
- **Usage**: The highest average Silhouette Score indicates the best-defined clusters, with scores ranging from -1 to +1.

## Conclusion
Both methods provide valuable insights for determining the optimal number of clusters and can be used together to ensure a comprehensive understanding of clustering performance.

# 

When you plot WCSS (or another measure of cluster compactness) against the number of clusters k, the resulting curve often resembles an arm with a bend (or "elbow").

Initially, as you increase k, WCSS decreases sharply, indicating that adding more clusters leads to significant reductions in variance within clusters.
Point of Diminishing Returns:

After a certain point (the "elbow"), the decrease in WCSS becomes less pronounced. This suggests that adding more clusters does not provide substantial benefits in terms of reducing variance.

The "elbow" represents a point where the model begins to exhibit diminishing returns, making it a natural stopping point for choosing the number of clusters.




