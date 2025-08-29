# Module 5: Unsupervised Learning 🔍

[← Previous Module](04_supervised_learning.md) | [Back to Main](../README.md) | [Next Module →](06_neural_networks.md)

## 📋 Table of Contents
1. [K-Means Clustering](#k-means-clustering)
2. [Hierarchical Clustering](#hierarchical-clustering)
3. [DBSCAN](#dbscan)
4. [Gaussian Mixture Models](#gaussian-mixture-models)
5. [Principal Component Analysis (PCA)](#principal-component-analysis)
6. [t-SNE and UMAP](#t-sne-and-umap)
7. [Anomaly Detection](#anomaly-detection)
8. [Practical Applications](#practical-applications)

---

## Overview

This module explores unsupervised learning techniques for discovering patterns and structure in unlabeled data. You'll learn clustering algorithms, dimensionality reduction methods, and anomaly detection approaches.

## Key Topics Covered

### Clustering Algorithms

#### K-Means Clustering
- Lloyd's algorithm
- K-means++ initialization
- Choosing optimal K (elbow method, silhouette score)
- Mini-batch K-means
- Limitations and assumptions

#### Hierarchical Clustering
- Agglomerative clustering
- Divisive clustering
- Linkage criteria (single, complete, average, Ward)
- Dendrograms
- Cutting the tree

#### DBSCAN
- Density-based clustering
- Core points, border points, and noise
- Epsilon and MinPts parameters
- Advantages over K-means
- OPTICS and HDBSCAN variants

#### Gaussian Mixture Models
- Expectation-Maximization (EM) algorithm
- Soft clustering
- Model selection with BIC/AIC
- Relationship to K-means

### Dimensionality Reduction

#### Principal Component Analysis (PCA)
- Eigendecomposition approach
- SVD approach
- Explained variance
- Choosing number of components
- Kernel PCA

#### t-SNE
- Preserving local structure
- Perplexity parameter
- Computational considerations
- Interpretation guidelines

#### Other Methods
- UMAP
- Autoencoders (preview)
- Independent Component Analysis (ICA)
- Non-negative Matrix Factorization (NMF)

### Anomaly Detection
- Statistical methods
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
- Applications in fraud detection

## Prerequisites
- Completed Modules 1-4
- Understanding of probability and statistics
- Linear algebra (especially for PCA)

## Learning Outcomes
By the end of this module, you will be able to:
- Apply clustering algorithms to segment data
- Reduce dimensionality while preserving information
- Detect anomalies and outliers
- Visualize high-dimensional data
- Choose appropriate unsupervised methods for different tasks

---


[**Continue to Module 6: Neural Networks and Deep Learning Basics →**](06_neural_networks.md)