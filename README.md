# Cryptocurrencies

# Overview
The purpose of this project is to use unsupervised machine learning techniques to analyze cryptocurrency data. The original cryptocurrency data from [CryptoCompare](https://min-api.cryptocompare.com/data/all/coinlist) is preprocessed using Pandas to fit Unsupervised Machine Learning models. A clustering algorithm is used to group data and hvPlot visualization are used to create a report that includes the cryptocurrencies currently on the trading market and how they could be grouped to create a classification system for the new investment.

# Resources
* Data Source: [crypto_data.csv](https://github.com/ramya-ramamur/Cryptocurrencies/tree/main/Resources)
* Software: Python 3.7.7, Anaconda Navigator 1.9.12, Conda 4.8.4, Jupyter Notebook 6.0.3


# Analysis Method
* Preprocessing the database : Used pandas to reduce dataset of 1,252 cryptocurrencies to 532 that could be used for machine learning.
  - Remove non active cryptocurrencies and cryptocurrencies that doesn't have an algorithm
  - Remove Trading status column, incomplete Data cryptocurrencies, any cryptocurrencies that hasn't been mined
  - Extract Coin Name out and hold separately
  - Use get_dummies method to distinguish algorithms into own feature
  - Scale data for proper weight
* Use Principle Component Analysis (PCA) - SKLearn to reduce the scaled data to three components for three dimensional modeling to thin down meaningful components. 
* Clustering cryptocurrencies using K-Means (Use elbow curve to best K value for K-means method)
* Visualizing classification results with 2D and 3D scatter plots.


Conclusions

Bitcoin is in a class by itself. This is obvious in both the three dimensional cluster modeling and the scatterplot. This is obviously useful as Bitcoin can be studied as an exemplar.
It would be very instrumental to run this analysis again without the Bitcoin data, so see if the elbow curve finds a different optimal cluster value and so that we can see the distribution of the data better in the three dimensional plot. Essentially there are two classes, Bitcoin and all others.
We can use these findings to compare all cryptocurrencies against Bitcoin, and to re-run the unsupervised machine learning without Bitcoin to more fully understand the rest of the cryptocurrency market.
