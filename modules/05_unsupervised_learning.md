# Module 5: Unsupervised Learning 🔍

[← Previous Module](04_supervised_learning.md) | [Back to Main](../README.md) | [Next Module →](06_neural_networks.md)

## 📋 Table of Contents
1. [Introduction to Unsupervised Learning](#introduction-to-unsupervised-learning)
2. [K-Means Clustering](#k-means-clustering)
3. [Hierarchical Clustering](#hierarchical-clustering)
4. [DBSCAN](#dbscan)
5. [Gaussian Mixture Models](#gaussian-mixture-models)
6. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
7. [t-SNE and UMAP](#t-sne-and-umap)
8. [Anomaly Detection](#anomaly-detection)
9. [Practical Implementation](#practical-implementation)
10. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

---

## Introduction to Unsupervised Learning

Unsupervised learning is about finding hidden patterns in data without labeled examples. It's like being a detective who discovers structure and relationships without being told what to look for.

### Key Characteristics
- **No labels**: Work with unlabeled data
- **Pattern discovery**: Find natural groupings and structures
- **Exploratory**: Often used for data exploration and understanding
- **Applications**:
  - Customer segmentation
  - Anomaly detection
  - Dimensionality reduction
  - Data compression

### Types of Unsupervised Learning

```
Unsupervised Learning
├── Clustering (grouping similar data)
│   ├── K-Means
│   ├── Hierarchical
│   ├── DBSCAN
│   └── GMM
├── Dimensionality Reduction (compress features)
│   ├── PCA
│   ├── t-SNE
│   └── UMAP
└── Anomaly Detection (find outliers)
    ├── Isolation Forest
    └── One-Class SVM
```

---

## K-Means Clustering

### Intuition
K-Means groups data into K clusters by minimizing the distance between points and their cluster centers.

### How It Works

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                       cluster_std=0.60, random_state=42)

# Apply K-means
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualize results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.title('True Clusters')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='*', s=300, c='red', edgecolor='black', linewidth=2)
plt.title('K-Means Clustering')
plt.show()
```

### Implementing K-Means from Scratch

```python
class KMeansScratch:
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)

        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            clusters = self._assign_clusters(X)

            # Update centroids
            new_centroids = self._update_centroids(X, clusters)

            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.labels_ = clusters
        return self

    def _assign_clusters(self, X):
        distances = np.zeros((len(X), self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, clusters):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
        return centroids

    def predict(self, X):
        return self._assign_clusters(X)
```

### Choosing Optimal K

```python
from sklearn.metrics import silhouette_score

def find_optimal_k(X, k_range=range(2, 11)):
    """Find optimal k using elbow method and silhouette score"""

    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow method
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True)

    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Find best k
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"Best k by silhouette score: {best_k}")

    return inertias, silhouette_scores
```

### K-Means++ Initialization

```python
def kmeans_plusplus(X, n_clusters, random_state=42):
    """K-means++ initialization for better starting centroids"""
    np.random.seed(random_state)

    # Choose first centroid randomly
    centroids = [X[np.random.randint(len(X))]]

    for _ in range(1, n_clusters):
        # Calculate distances to nearest centroid
        distances = np.array([min([np.linalg.norm(x - c) for c in centroids])
                              for x in X])

        # Choose next centroid with probability proportional to distance squared
        probabilities = distances ** 2
        probabilities /= probabilities.sum()

        cumulative_probs = probabilities.cumsum()
        r = np.random.rand()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                centroids.append(X[j])
                break

    return np.array(centroids)
```

### Mini-Batch K-Means for Large Datasets

```python
from sklearn.cluster import MiniBatchKMeans

# For large datasets
n_samples = 10000
X_large = np.random.randn(n_samples, 50)  # 50 features

# Standard K-means (slower)
kmeans = KMeans(n_clusters=10, random_state=42)

# Mini-batch K-means (faster)
mb_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=100, random_state=42)

# Time comparison
import time

start = time.time()
kmeans.fit(X_large)
print(f"K-means time: {time.time() - start:.2f}s")

start = time.time()
mb_kmeans.fit(X_large)
print(f"Mini-batch K-means time: {time.time() - start:.2f}s")
```

---

## Hierarchical Clustering

### Concept
Builds a hierarchy of clusters either bottom-up (agglomerative) or top-down (divisive).

### Agglomerative Clustering Implementation

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# Perform hierarchical clustering
linkage_matrix = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Apply clustering with specific number of clusters
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clustering.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.title('Agglomerative Clustering Results')
plt.show()
```

### Different Linkage Criteria

```python
def compare_linkage_methods(X):
    """Compare different linkage criteria"""

    linkage_methods = ['single', 'complete', 'average', 'ward']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, method in enumerate(linkage_methods):
        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=3, linkage=method)
        labels = clustering.fit_predict(X)

        # Plot
        axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
        axes[idx].set_title(f'Linkage: {method}')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

# Test with different data shapes
X_blob, _ = make_blobs(n_samples=150, centers=3, random_state=42)
compare_linkage_methods(X_blob)
```

### Cutting the Dendrogram

```python
from scipy.cluster.hierarchy import fcluster

def cut_dendrogram(X, linkage_matrix, threshold=None, n_clusters=None):
    """Cut dendrogram at specific height or to get n clusters"""

    if n_clusters:
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    elif threshold:
        labels = fcluster(linkage_matrix, threshold, criterion='distance')
    else:
        raise ValueError("Specify either threshold or n_clusters")

    return labels

# Example usage
linkage_matrix = linkage(X, method='ward')
labels_by_n = cut_dendrogram(X, linkage_matrix, n_clusters=3)
labels_by_dist = cut_dendrogram(X, linkage_matrix, threshold=10)

print(f"Unique clusters by n_clusters: {np.unique(labels_by_n)}")
print(f"Unique clusters by threshold: {np.unique(labels_by_dist)}")
```

---

## DBSCAN

### Density-Based Spatial Clustering
DBSCAN finds clusters of arbitrary shape based on density.

### Key Concepts
- **Core points**: Have at least MinPts within eps radius
- **Border points**: Within eps of core point but not core themselves
- **Noise points**: Neither core nor border

### Implementation

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Generate data with noise
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Visualize
plt.figure(figsize=(10, 6))
unique_labels = set(clusters)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black for noise
        col = 'black'
        marker_size = 20
    else:
        marker_size = 50

    class_members = (clusters == k)
    plt.scatter(X[class_members, 0], X[class_members, 1],
               s=marker_size, c=[col], alpha=0.6,
               label=f'Cluster {k}' if k != -1 else 'Noise')

plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Print statistics
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print(f'Number of clusters: {n_clusters}')
print(f'Number of noise points: {n_noise}')
```

### Finding Optimal Parameters

```python
from sklearn.neighbors import NearestNeighbors

def find_optimal_eps(X, min_samples=5):
    """Find optimal eps using k-distance plot"""

    # Calculate k-distances
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)

    # Sort distances
    distances = np.sort(distances[:, -1])

    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.ylabel(f'{min_samples}-NN Distance')
    plt.xlabel('Points sorted by distance')
    plt.title('K-distance Graph for Epsilon Selection')
    plt.grid(True)
    plt.show()

    # Find elbow (you might need to adjust this)
    # Simple approach: look for maximum curvature
    diffs = np.diff(distances)
    diffs2 = np.diff(diffs)
    elbow_idx = np.argmax(diffs2) + 2
    optimal_eps = distances[elbow_idx]

    print(f"Suggested eps: {optimal_eps:.3f}")
    return optimal_eps

# Example usage
optimal_eps = find_optimal_eps(X_scaled, min_samples=5)
```

### HDBSCAN - Hierarchical DBSCAN

```python
# Install: pip install hdbscan
import hdbscan

# HDBSCAN automatically finds varying densities
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
cluster_labels = clusterer.fit_predict(X_scaled)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.title('HDBSCAN Clustering')
plt.colorbar()
plt.show()

# Cluster persistence (stability)
clusterer.condensed_tree_.plot(select_clusters=True)
plt.show()
```

---

## Gaussian Mixture Models

### Soft Clustering with Probabilities
GMM assumes data comes from a mixture of Gaussian distributions.

### Basic Implementation

```python
from sklearn.mixture import GaussianMixture

# Generate data
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2,
                       cluster_std=1.0, random_state=42)

# Fit GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Predict clusters (hard assignment)
labels = gmm.predict(X)

# Get probabilities (soft assignment)
probs = gmm.predict_proba(X)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Hard clustering
ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
ax1.set_title('GMM Hard Clustering')

# Soft clustering (show uncertainty)
uncertainty = 1 - np.max(probs, axis=1)
scatter = ax2.scatter(X[:, 0], X[:, 1], c=uncertainty, cmap='RdYlGn_r', alpha=0.6)
ax2.set_title('GMM Uncertainty (red = high uncertainty)')
plt.colorbar(scatter, ax=ax2)

plt.tight_layout()
plt.show()
```

### Model Selection with BIC/AIC

```python
def select_gmm_components(X, n_components_range=range(1, 11)):
    """Select optimal number of components using BIC and AIC"""

    bic_scores = []
    aic_scores = []

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    # Plot scores
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(n_components_range, bic_scores, 'bo-')
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('BIC Score')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(n_components_range, aic_scores, 'ro-')
    plt.xlabel('Number of components')
    plt.ylabel('AIC')
    plt.title('AIC Score')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Find optimal
    best_n_bic = n_components_range[np.argmin(bic_scores)]
    best_n_aic = n_components_range[np.argmin(aic_scores)]

    print(f"Best n_components by BIC: {best_n_bic}")
    print(f"Best n_components by AIC: {best_n_aic}")

    return bic_scores, aic_scores

# Example usage
bic, aic = select_gmm_components(X)
```

### Implementing EM Algorithm from Scratch

```python
class GMMScratch:
    def __init__(self, n_components=3, max_iters=100, tol=1e-4):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = [np.eye(n_features) for _ in range(self.n_components)]

        log_likelihood = 0

        for iteration in range(self.max_iters):
            # E-step: Calculate responsibilities
            responsibilities = self._e_step(X)

            # M-step: Update parameters
            self._m_step(X, responsibilities)

            # Check convergence
            new_log_likelihood = self._compute_log_likelihood(X)
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

        return self

    def _e_step(self, X):
        """Calculate responsibilities (posterior probabilities)"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self._multivariate_gaussian(
                X, self.means[k], self.covariances[k])

        # Normalize
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X, responsibilities):
        """Update parameters"""
        n_samples = X.shape[0]

        for k in range(self.n_components):
            responsibility_k = responsibilities[:, k]
            sum_resp = responsibility_k.sum()

            # Update weight
            self.weights[k] = sum_resp / n_samples

            # Update mean
            self.means[k] = (responsibility_k @ X) / sum_resp

            # Update covariance
            diff = X - self.means[k]
            self.covariances[k] = (responsibility_k * diff.T) @ diff / sum_resp

    def _multivariate_gaussian(self, X, mean, cov):
        """Calculate multivariate Gaussian probability"""
        n = X.shape[1]
        diff = X - mean
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        norm_const = 1.0 / np.sqrt((2 * np.pi) ** n * det_cov)
        exp_part = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))

        return norm_const * exp_part

    def _compute_log_likelihood(self, X):
        """Compute log likelihood"""
        likelihood = np.zeros(X.shape[0])
        for k in range(self.n_components):
            likelihood += self.weights[k] * self._multivariate_gaussian(
                X, self.means[k], self.covariances[k])
        return np.log(likelihood).sum()
```

---

## Principal Component Analysis (PCA)

### Dimensionality Reduction
PCA finds the directions of maximum variance in high-dimensional data.

### Basic PCA Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate high-dimensional data
np.random.seed(42)
n_samples = 500
n_features = 50
X = np.random.randn(n_samples, n_features)

# Add some structure
for i in range(5):
    X[:, i*10:(i+1)*10] += np.random.randn(n_samples, 1) * 5

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='k', linestyle='--', label='95% Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Find number of components for 95% variance
n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")
```

### PCA from Scratch

```python
class PCAScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store components
        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance = eigenvalues[:self.n_components]

        # Calculate explained variance ratio
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = self.explained_variance / total_variance

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca):
        return X_pca @ self.components + self.mean
```

### PCA for Visualization

```python
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Apply PCA for 2D visualization
pca_2d = PCA(n_components=2)
X_iris_2d = pca_2d.fit_transform(X_iris)

# Apply PCA for 3D visualization
pca_3d = PCA(n_components=3)
X_iris_3d = pca_3d.fit_transform(X_iris)

# Plot
fig = plt.figure(figsize=(14, 6))

# 2D plot
ax1 = fig.add_subplot(121)
for i, target_name in enumerate(iris.target_names):
    ax1.scatter(X_iris_2d[y_iris == i, 0], X_iris_2d[y_iris == i, 1],
               label=target_name, alpha=0.7)
ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
ax1.set_title('PCA - 2D Projection')
ax1.legend()
ax1.grid(True)

# 3D plot
ax2 = fig.add_subplot(122, projection='3d')
for i, target_name in enumerate(iris.target_names):
    ax2.scatter(X_iris_3d[y_iris == i, 0],
               X_iris_3d[y_iris == i, 1],
               X_iris_3d[y_iris == i, 2],
               label=target_name, alpha=0.7)
ax2.set_xlabel(f'PC1')
ax2.set_ylabel(f'PC2')
ax2.set_zlabel(f'PC3')
ax2.set_title('PCA - 3D Projection')
ax2.legend()

plt.tight_layout()
plt.show()
```

### Kernel PCA for Non-linear Data

```python
from sklearn.decomposition import KernelPCA

# Generate non-linear data
from sklearn.datasets import make_circles
X_circles, y_circles = make_circles(n_samples=400, factor=0.3, noise=0.05)

# Compare linear PCA and kernel PCA
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original data
axes[0, 0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis')
axes[0, 0].set_title('Original Data')

# Linear PCA
pca_linear = PCA(n_components=2)
X_pca_linear = pca_linear.fit_transform(X_circles)
axes[0, 1].scatter(X_pca_linear[:, 0], X_pca_linear[:, 1], c=y_circles, cmap='viridis')
axes[0, 1].set_title('Linear PCA')

# Kernel PCA with different kernels
kernels = ['rbf', 'poly', 'sigmoid', 'cosine']
positions = [(0, 2), (1, 0), (1, 1), (1, 2)]

for kernel, (i, j) in zip(kernels, positions):
    kpca = KernelPCA(n_components=2, kernel=kernel, gamma=15)
    X_kpca = kpca.fit_transform(X_circles)
    axes[i, j].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y_circles, cmap='viridis')
    axes[i, j].set_title(f'Kernel PCA ({kernel})')

plt.tight_layout()
plt.show()
```

---

## t-SNE and UMAP

### t-SNE: Preserving Local Structure

```python
from sklearn.manifold import TSNE

# Generate high-dimensional clustered data
n_samples = 1000
n_features = 50
n_clusters = 5

# Create clustered data in high dimensions
X_high = []
y_high = []
for i in range(n_clusters):
    center = np.random.randn(n_features) * 10
    cluster_data = center + np.random.randn(n_samples // n_clusters, n_features)
    X_high.append(cluster_data)
    y_high.extend([i] * (n_samples // n_clusters))

X_high = np.vstack(X_high)
y_high = np.array(y_high)

# Apply t-SNE with different perplexities
perplexities = [5, 30, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_high)

    axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_high, cmap='tab10', alpha=0.6)
    axes[idx].set_title(f't-SNE (perplexity={perplexity})')
    axes[idx].set_xlabel('t-SNE 1')
    axes[idx].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

### UMAP: Faster Alternative to t-SNE

```python
# Install: pip install umap-learn
import umap

# Apply UMAP
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(X_high)

# Compare t-SNE and UMAP
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_high)
ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_high, cmap='tab10', alpha=0.6)
ax1.set_title('t-SNE')
ax1.set_xlabel('Component 1')
ax1.set_ylabel('Component 2')

# UMAP
ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y_high, cmap='tab10', alpha=0.6)
ax2.set_title('UMAP')
ax2.set_xlabel('Component 1')
ax2.set_ylabel('Component 2')

plt.tight_layout()
plt.show()

# Time comparison
import time

start = time.time()
_ = TSNE(n_components=2).fit_transform(X_high)
tsne_time = time.time() - start

start = time.time()
_ = umap.UMAP(n_components=2).fit_transform(X_high)
umap_time = time.time() - start

print(f"t-SNE time: {tsne_time:.2f}s")
print(f"UMAP time: {umap_time:.2f}s")
print(f"UMAP is {tsne_time/umap_time:.1f}x faster")
```

### Combining PCA with t-SNE

```python
def pca_then_tsne(X, n_pca_components=50, n_tsne_components=2):
    """Apply PCA before t-SNE for better performance"""

    # First reduce with PCA
    pca = PCA(n_components=min(n_pca_components, X.shape[1]))
    X_pca = pca.fit_transform(X)

    print(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} dimensions with PCA")
    print(f"Retained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Then apply t-SNE
    tsne = TSNE(n_components=n_tsne_components, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    return X_tsne, pca, tsne

# Example with high-dimensional data
X_reduced, pca_model, tsne_model = pca_then_tsne(X_high, n_pca_components=30)

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_high, cmap='tab10', alpha=0.6)
plt.title('PCA + t-SNE')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar()
plt.show()
```

---

## Anomaly Detection

### Statistical Methods

```python
from scipy import stats

def detect_outliers_zscore(X, threshold=3):
    """Detect outliers using Z-score"""
    z_scores = np.abs(stats.zscore(X))
    outliers = (z_scores > threshold).any(axis=1)
    return outliers

def detect_outliers_iqr(X, k=1.5):
    """Detect outliers using IQR method"""
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1

    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    outliers = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
    return outliers

# Generate data with outliers
np.random.seed(42)
X_normal = np.random.randn(100, 2)
X_outliers = np.random.uniform(-6, 6, (10, 2))
X_with_outliers = np.vstack([X_normal, X_outliers])
y_true = np.array([0] * 100 + [1] * 10)

# Detect outliers
outliers_zscore = detect_outliers_zscore(X_with_outliers)
outliers_iqr = detect_outliers_iqr(X_with_outliers)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(X_with_outliers[~outliers_zscore, 0], X_with_outliers[~outliers_zscore, 1],
           c='blue', label='Normal', alpha=0.6)
ax1.scatter(X_with_outliers[outliers_zscore, 0], X_with_outliers[outliers_zscore, 1],
           c='red', label='Outlier', marker='x', s=100)
ax1.set_title('Z-Score Method')
ax1.legend()

ax2.scatter(X_with_outliers[~outliers_iqr, 0], X_with_outliers[~outliers_iqr, 1],
           c='blue', label='Normal', alpha=0.6)
ax2.scatter(X_with_outliers[outliers_iqr, 0], X_with_outliers[outliers_iqr, 1],
           c='red', label='Outlier', marker='x', s=100)
ax2.set_title('IQR Method')
ax2.legend()

plt.tight_layout()
plt.show()
```

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
predictions = iso_forest.fit_predict(X_with_outliers)
outliers_iso = predictions == -1

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_with_outliers[~outliers_iso, 0], X_with_outliers[~outliers_iso, 1],
           c='blue', label='Normal', alpha=0.6)
plt.scatter(X_with_outliers[outliers_iso, 0], X_with_outliers[outliers_iso, 1],
           c='red', label='Anomaly', marker='x', s=100)
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Evaluate
from sklearn.metrics import classification_report
print("Isolation Forest Performance:")
print(classification_report(y_true, outliers_iso.astype(int)))
```

### One-Class SVM

```python
from sklearn.svm import OneClassSVM

# Train One-Class SVM
ocsvm = OneClassSVM(gamma='auto', nu=0.1)
ocsvm.fit(X_normal)  # Train only on normal data
predictions = ocsvm.predict(X_with_outliers)
outliers_ocsvm = predictions == -1

# Visualize decision boundary
xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Blues_r', alpha=0.3)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.scatter(X_with_outliers[~outliers_ocsvm, 0], X_with_outliers[~outliers_ocsvm, 1],
           c='blue', label='Normal', alpha=0.6)
plt.scatter(X_with_outliers[outliers_ocsvm, 0], X_with_outliers[outliers_ocsvm, 1],
           c='red', label='Anomaly', marker='x', s=100)
plt.title('One-Class SVM Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

### Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

# Train LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
predictions = lof.fit_predict(X_with_outliers)
outliers_lof = predictions == -1

# Get anomaly scores
lof_scores = lof.negative_outlier_factor_

# Visualize with scores
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_with_outliers[:, 0], X_with_outliers[:, 1],
                     c=lof_scores, cmap='coolwarm', s=50, alpha=0.6)
plt.colorbar(scatter, label='LOF Score')
plt.scatter(X_with_outliers[outliers_lof, 0], X_with_outliers[outliers_lof, 1],
           edgecolors='red', facecolors='none', s=100, linewidths=2)
plt.title('Local Outlier Factor')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

---

## Practical Implementation

### Complete Unsupervised Learning Pipeline

```python
class UnsupervisedPipeline:
    def __init__(self, scaling=True, dim_reduction='pca', clustering='kmeans'):
        self.scaling = scaling
        self.dim_reduction = dim_reduction
        self.clustering = clustering
        self.scaler = None
        self.reducer = None
        self.clusterer = None

    def fit(self, X):
        # Scaling
        if self.scaling:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Dimensionality reduction
        if self.dim_reduction == 'pca':
            self.reducer = PCA(n_components=0.95)  # Keep 95% variance
            X_reduced = self.reducer.fit_transform(X)
        elif self.dim_reduction == 'tsne':
            self.reducer = TSNE(n_components=2)
            X_reduced = self.reducer.fit_transform(X)
        elif self.dim_reduction == 'umap':
            self.reducer = umap.UMAP(n_components=2)
            X_reduced = self.reducer.fit_transform(X)
        else:
            X_reduced = X

        # Clustering
        if self.clustering == 'kmeans':
            self.clusterer = KMeans(n_clusters=3)
        elif self.clustering == 'dbscan':
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        elif self.clustering == 'gmm':
            self.clusterer = GaussianMixture(n_components=3)

        self.labels_ = self.clusterer.fit_predict(X_reduced)
        return self

    def transform(self, X):
        if self.scaling and self.scaler:
            X = self.scaler.transform(X)
        if self.reducer:
            X = self.reducer.transform(X)
        return X

    def predict(self, X):
        X_transformed = self.transform(X)
        return self.clusterer.predict(X_transformed)
```

### Real-world Example: Customer Segmentation

```python
import pandas as pd

def create_customer_data():
    """Create synthetic customer data"""
    np.random.seed(42)
    n_customers = 1000

    # Create customer segments
    segments = {
        'high_value': {
            'size': 200,
            'spending': (1000, 200),
            'frequency': (20, 3),
            'recency': (5, 2)
        },
        'regular': {
            'size': 500,
            'spending': (300, 100),
            'frequency': (10, 3),
            'recency': (20, 5)
        },
        'low_value': {
            'size': 300,
            'spending': (50, 30),
            'frequency': (3, 2),
            'recency': (60, 15)
        }
    }

    data = []
    true_labels = []

    for i, (segment_name, params) in enumerate(segments.items()):
        segment_data = pd.DataFrame({
            'total_spending': np.random.normal(params['spending'][0],
                                              params['spending'][1],
                                              params['size']),
            'purchase_frequency': np.random.normal(params['frequency'][0],
                                                  params['frequency'][1],
                                                  params['size']),
            'days_since_last_purchase': np.random.normal(params['recency'][0],
                                                        params['recency'][1],
                                                        params['size']),
            'true_segment': segment_name
        })
        data.append(segment_data)
        true_labels.extend([i] * params['size'])

    customer_data = pd.concat(data, ignore_index=True)
    return customer_data, np.array(true_labels)

# Create and analyze customer data
customer_df, true_segments = create_customer_data()

# Prepare features
X_customers = customer_df[['total_spending', 'purchase_frequency', 'days_since_last_purchase']].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_customers)

# Find optimal number of clusters
silhouette_scores = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette Score = {score:.3f}")

best_k = range(2, 8)[np.argmax(silhouette_scores)]
print(f"\nBest k: {best_k}")

# Apply final clustering
final_kmeans = KMeans(n_clusters=best_k, random_state=42)
customer_segments = final_kmeans.fit_predict(X_scaled)

# Analyze segments
segment_analysis = pd.DataFrame(X_customers,
                               columns=['spending', 'frequency', 'recency'])
segment_analysis['segment'] = customer_segments

print("\nSegment Statistics:")
print(segment_analysis.groupby('segment').agg({
    'spending': ['mean', 'std', 'count'],
    'frequency': 'mean',
    'recency': 'mean'
}).round(2))

# Visualize segments in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X_customers[:, 0], X_customers[:, 1], X_customers[:, 2],
                    c=customer_segments, cmap='viridis', s=50, alpha=0.6)

ax.set_xlabel('Total Spending')
ax.set_ylabel('Purchase Frequency')
ax.set_zlabel('Days Since Last Purchase')
ax.set_title('Customer Segmentation')

plt.colorbar(scatter, label='Segment')
plt.show()

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=customer_segments,
                     cmap='viridis', s=50, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('Customer Segments (PCA Projection)')
plt.colorbar(scatter, label='Segment')
plt.show()
```

---

## Common Pitfalls & Solutions

### 1. Not Scaling Features
```python
# WRONG: Clustering without scaling
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)  # Features have different scales!

# CORRECT: Scale first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X_scaled)
```

### 2. Choosing Wrong Number of Clusters
```python
def comprehensive_cluster_evaluation(X, max_k=10):
    """Evaluate clustering with multiple metrics"""

    metrics = {
        'silhouette': [],
        'calinski': [],
        'davies_bouldin': []
    }

    K = range(2, max_k + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        metrics['silhouette'].append(silhouette_score(X, labels))
        metrics['calinski'].append(metrics.calinski_harabasz_score(X, labels))
        metrics['davies_bouldin'].append(metrics.davies_bouldin_score(X, labels))

    # Plot all metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(K, metrics['silhouette'], 'bo-')
    axes[0].set_title('Silhouette Score (higher is better)')
    axes[0].set_xlabel('k')

    axes[1].plot(K, metrics['calinski'], 'go-')
    axes[1].set_title('Calinski-Harabasz Score (higher is better)')
    axes[1].set_xlabel('k')

    axes[2].plot(K, metrics['davies_bouldin'], 'ro-')
    axes[2].set_title('Davies-Bouldin Score (lower is better)')
    axes[2].set_xlabel('k')

    plt.tight_layout()
    plt.show()

    return metrics
```

### 3. Ignoring Cluster Assumptions
```python
def check_clustering_assumptions(X, method='kmeans'):
    """Check if data meets clustering assumptions"""

    if method == 'kmeans':
        # K-means assumes spherical clusters
        # Check if clusters are roughly spherical using Hopkins statistic
        from sklearn.neighbors import NearestNeighbors

        n = len(X)
        m = int(0.1 * n)  # Sample size

        # Random sampling
        random_indices = np.random.choice(n, m, replace=False)
        X_sample = X[random_indices]

        # Calculate nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=2).fit(X)
        distances, _ = nbrs.kneighbors(X_sample)
        u_distances = distances[:, 1]

        # Random points from data space
        X_random = np.random.uniform(X.min(axis=0), X.max(axis=0), (m, X.shape[1]))
        nbrs_random = NearestNeighbors(n_neighbors=1).fit(X)
        w_distances, _ = nbrs_random.kneighbors(X_random)
        w_distances = w_distances.flatten()

        # Hopkins statistic
        hopkins = np.sum(w_distances) / (np.sum(u_distances) + np.sum(w_distances))

        print(f"Hopkins statistic: {hopkins:.3f}")
        if hopkins < 0.3:
            print("Data has clustering tendency (good for k-means)")
        elif hopkins > 0.7:
            print("Data is uniformly distributed (not ideal for k-means)")
        else:
            print("Data is random (clustering may not be meaningful)")

        return hopkins
```

### 4. Wrong Distance Metric
```python
from sklearn.metrics import pairwise_distances

def compare_distance_metrics(X, metrics=['euclidean', 'manhattan', 'cosine']):
    """Compare different distance metrics for clustering"""

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

    for idx, metric in enumerate(metrics):
        # Calculate pairwise distances
        distances = pairwise_distances(X, metric=metric)

        # Perform clustering with custom metric
        if metric == 'euclidean':
            clusterer = KMeans(n_clusters=3)
            labels = clusterer.fit_predict(X)
        else:
            # Use Agglomerative clustering for custom metrics
            clusterer = AgglomerativeClustering(n_clusters=3,
                                               affinity=metric,
                                               linkage='average')
            labels = clusterer.fit_predict(X)

        # Visualize
        axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        axes[idx].set_title(f'{metric.capitalize()} Distance')

    plt.tight_layout()
    plt.show()
```

### 5. Misinterpreting t-SNE
```python
def tsne_stability_check(X, n_runs=5):
    """Check t-SNE stability across multiple runs"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i in range(min(n_runs, 6)):
        tsne = TSNE(n_components=2, random_state=i)
        X_embedded = tsne.fit_transform(X)

        axes[i].scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.6)
        axes[i].set_title(f't-SNE Run {i+1}')
        axes[i].set_xlabel('Component 1')
        axes[i].set_ylabel('Component 2')

    # Hide extra subplots
    for i in range(n_runs, 6):
        axes[i].set_visible(False)

    plt.suptitle('t-SNE Stability Check - Different Random Seeds')
    plt.tight_layout()
    plt.show()

    print("Note: t-SNE preserves local structure but global positions may vary")
```

---

## Practice Exercises

### Exercise 1: Implement DBSCAN from Scratch
```python
# TODO: Implement DBSCAN algorithm
# 1. Find core points
# 2. Build clusters from core points
# 3. Assign border points
# 4. Mark noise points
```

### Exercise 2: Create an Anomaly Detection Ensemble
```python
# TODO: Combine multiple anomaly detection methods
# 1. Use Isolation Forest, One-Class SVM, and LOF
# 2. Create voting mechanism
# 3. Evaluate on synthetic data with known anomalies
```

### Exercise 3: Dimension Reduction Comparison
```python
# TODO: Compare PCA, t-SNE, and UMAP on MNIST digits
# 1. Load subset of MNIST
# 2. Apply each method
# 3. Measure preservation of cluster structure
# 4. Compare computation time
```

### Exercise 4: Hierarchical GMM
```python
# TODO: Implement hierarchical clustering with GMM
# 1. Start with one cluster
# 2. Recursively split using GMM
# 3. Use BIC to decide when to stop splitting
```

---

## Summary and Key Takeaways

### Unsupervised Learning Essentials
1. **K-Means**: Fast, simple, assumes spherical clusters
2. **DBSCAN**: Finds arbitrary shapes, handles noise
3. **Hierarchical**: Provides dendrogram, no preset k
4. **GMM**: Soft clustering with probabilities
5. **PCA**: Linear dimensionality reduction
6. **t-SNE/UMAP**: Non-linear visualization
7. **Anomaly Detection**: Multiple approaches for outliers

### Best Practices Checklist
- [ ] Always scale features before clustering
- [ ] Use multiple metrics to evaluate clusters
- [ ] Visualize data before choosing algorithm
- [ ] Check clustering assumptions
- [ ] Validate stability of results
- [ ] Consider computational complexity
- [ ] Use domain knowledge to interpret results
- [ ] Combine multiple methods when appropriate
- [ ] Document parameter choices
- [ ] Test on synthetic data first

### When to Use What?

| Method | Use When | Avoid When |
|--------|----------|------------|
| K-Means | Spherical clusters, known k, large datasets | Non-globular clusters, outliers present |
| DBSCAN | Arbitrary shapes, noise present, unknown k | Varying densities, high dimensions |
| Hierarchical | Need dendrogram, small datasets | Very large datasets |
| GMM | Soft assignments needed, elliptical clusters | Non-Gaussian distributions |
| PCA | Linear relationships, feature reduction | Non-linear manifolds |
| t-SNE | Visualization, local structure important | Need reproducibility, large datasets |
| UMAP | Fast visualization, preserve global structure | Need exact distance preservation |

---

## Additional Resources

### Libraries to Master
- **scikit-learn**: Main unsupervised learning library
- **umap-learn**: UMAP implementation
- **hdbscan**: Hierarchical DBSCAN
- **pyclustering**: Additional clustering algorithms

### Recommended Projects
1. Customer segmentation system
2. Image compression using PCA
3. Anomaly detection for network traffic
4. Document clustering for topic discovery
5. Recommendation system using clustering

### Next Steps
- Module 6: Neural Networks and Deep Learning
- Advanced: Autoencoders for unsupervised learning
- Advanced: Self-organizing maps (SOM)
- Advanced: Spectral clustering

---

[**Continue to Module 6: Neural Networks and Deep Learning Basics →**](06_neural_networks.md)