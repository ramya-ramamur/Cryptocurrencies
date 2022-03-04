# Cryptocurrencies

# Overview
The purpose of this project is to use unsupervised machine learning techniques to analyze cryptocurrency data. The original cryptocurrency data from [CryptoCompare](https://min-api.cryptocompare.com/data/all/coinlist) is preprocessed using Pandas to fit Unsupervised Machine Learning models. A clustering algorithm is used to group data and hvPlot visualization are used to create a report that includes the cryptocurrencies currently on the trading market and how they could be grouped to create a classification system for the new investment.

# Resources
* Data Source: [crypto_data.csv](https://github.com/ramya-ramamur/Cryptocurrencies/tree/main/Resources)
* Software: Python 3.8.8, Pandas Dataframe, Jupyter Notebook 6.4.6, Anaconda Navigator 2.1.1, imbalanced-learn, skikit-learn

# Analysis & Results
### Preprocessing the database : Used pandas to reduce dataset of 1,252 cryptocurrencies to 532 that could be used for machine learning.
  - Remove non active cryptocurrencies and cryptocurrencies that doesn't have an algorithm
  - Remove Trading status column, incomplete Data cryptocurrencies, any cryptocurrencies that hasn't been mined
  - Extract Coin Name out and hold separately
  - Use get_dummies method to distinguish algorithms into own feature
  - Scale data for proper weight

![Screen Shot 2022-03-04 at 12 10 53 PM](https://user-images.githubusercontent.com/75961057/156834529-f4d626f5-fb45-43f6-84f5-891ae4a886f0.png)

### Use Principle Component Analysis (PCA) - SKLearn to reduce the scaled data to three components for three dimensional modeling to thin down meaningful components. 
PCA is a statistical technique to speed up machine learning algorithms when the number of input features (or dimensions) is too high. PCA reduces the number of dimensions by transforming a large set of variables into a smaller one that contains most of the information in the original large set.

![PCA](https://user-images.githubusercontent.com/75961057/156835152-d783e418-8666-49b4-8925-2056b0327fef.png)

### Clustering cryptocurrencies using K-Means (Use elbow curve to best K value(4) for K-means method)

The objective of K-means is to group similar data points together and discover underlying patterns. To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset. These clusters are then determined by the means of all the points that will belong to the cluster.

The K-means algorithm groups the data into K clusters, where belonging to a cluster is based on some similarity or distance measure to a centroid.
A centroid is a data point that is the arithmetic mean position of all the points on a cluster. The centroid is found by taking the mean of all the x values in a cluster, and the mean of all the y values in a cluster.

###### Elbow Curve
The best k value appears to be 4 so we used 4 clusters to categorize the crytocurrencies.

![Elbow Curve](https://user-images.githubusercontent.com/75961057/156834011-819b2904-38df-4475-9ed6-bb03f88657ef.png)

###### Dataframe after clustering.

![df_after_clustering_with_KMeans](https://user-images.githubusercontent.com/75961057/156836009-b649c7dc-a35b-42e2-a49e-69b2650dab3e.png)

### Visualizing Cryptocurrencies Results

#### 3D scatter plot with Plotly Express

![3d_scatter_plot_clusters](https://user-images.githubusercontent.com/75961057/156836236-629f3cbd-58ec-4417-8f46-4584a3298f28.png)

#### Table with tradable cryptocurrencies using the hvplot.table() function.

![hvplot_table_with_tradable_cryptocurrencies](https://user-images.githubusercontent.com/75961057/156836726-17501313-94d7-4a1b-a433-5a19418560ee.png)

#### A hvplot scatter plot with x="TotalCoinsMined", y="TotalCoinSupply", and by="Class"
For this we will first make a new DataFrame that has the scaled data with the clustered_df DataFrame index.

![Screen Shot 2022-03-04 at 12 32 27 PM](https://user-images.githubusercontent.com/75961057/156837404-cb513889-562f-4b8f-8a6e-a5f1940801e5.png)

![hvplot_scatter_totalcoinsmined_vs_totalcoinsupply](https://user-images.githubusercontent.com/75961057/156837102-c1a90e89-1724-4701-95bb-dd486e683360.png)

# Summary
Classification of 532 cryptocurrencies based on similarities of their features has resulted in dividing them into four classes. We can see that classes 3 and 1 are the most profitable cryptos with Bitcoin and Ethereum (class 3) being the most popular. The investment banks can now concentrate on the the individuality of these classes to determined their performance and decide the profitabilty of investing in them. 
