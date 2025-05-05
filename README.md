# Customer_segmentation
This project performs customer segmentation using unsupervised learning (KMeans clustering) and visualizes the results in an interactive Streamlit web app.

# Files repository:
customer_segmentation.ipynb: Jupyter Notebook with full data preprocessing, clustering, and visualizations.

app.py: Streamlit app script for interactive customer segmentation and cluster visualization.

customer_segmentation.csv: Sample dataset.


# Tech Stack:
Python
Pandas, NumPy
Scikit-learn (KMeans, StandardScaler)
Matplotlib, Seaborn
Streamlit

## 1 Data Loading

Loads the Mall_Customers.csv dataset.

Displays first few rows using head().

## 2.Data Preprocessing

Drops non-numeric or irrelevant columns like CustomerID.

Checks and handles missing values.

Uses StandardScaler to normalize features for clustering.

## 3.Exploratory Data Analysis (EDA)

Visualizes relationships between features like Age, Income, and Spending Score using scatter plots and pairplots.

Shows distributions and correlation heatmaps for better understanding of data.

## 4.Elbow Method

Uses the Within-Cluster Sum of Squares (WCSS) to determine the optimal number of clusters (k) for KMeans.

Plots a line graph to visualize the elbow point.

## 5.KMeans Clustering

Applies the KMeans algorithm to the scaled data.

Adds cluster labels back to the original dataset for interpretation.

## 6.Cluster Analysis

Descriptive statistics for each cluster group.

Visualizes clusters using:

Scatter plots

Cluster centroids

2D PCA-based visualization

