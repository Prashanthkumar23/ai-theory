# Module 1: Mathematical Foundations - Linear Algebra 📐

[← Back to Main](../README.md) | [Next Module →](02_probability_statistics.md)

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Scalars, Vectors, and Matrices](#scalars-vectors-and-matrices)
3. [Matrix Operations](#matrix-operations)
4. [Special Matrices](#special-matrices)
5. [Linear Transformations](#linear-transformations)
6. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
7. [Applications in Machine Learning](#applications-in-machine-learning)
8. [Practical Exercises](#practical-exercises)
9. [Summary and Key Takeaways](#summary-and-key-takeaways)

---

## Introduction

Linear algebra is the backbone of machine learning and AI. Almost every algorithm in machine learning can be expressed in terms of linear algebra operations. This module will give you a solid foundation in the essential concepts.

![Linear Algebra in ML](https://miro.medium.com/max/1400/1*YWUar8sNlhmDFKBW0vDmgg.png)

### Why Linear Algebra for AI?

- **Data Representation**: Data is represented as vectors and matrices
- **Transformations**: Neural networks perform linear transformations
- **Optimization**: Gradient descent uses vector calculus
- **Dimensionality Reduction**: PCA relies on eigendecomposition
- **Computer Vision**: Images are matrices of pixels

---

## Scalars, Vectors, and Matrices

### 1. Scalars

A **scalar** is a single number. In machine learning, scalars often represent:
- Learning rates (α = 0.01)
- Regularization parameters (λ = 0.1)
- Individual predictions or errors

**Notation**: We denote scalars with lowercase letters: *a*, *b*, *x*, *y*

### 2. Vectors

A **vector** is an ordered array of numbers. Think of it as a list of features.

![Vector Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Vector_from_A_to_B.svg/1200px-Vector_from_A_to_B.svg.png)

**Mathematical Representation:**
```
x = [x₁]
    [x₂]
    [x₃]
    [...]
    [xₙ]
```

**In Machine Learning:**
- Feature vectors: `[age, height, weight, income]`
- Word embeddings: 300-dimensional vectors representing words
- Image pixels: Flattened into a vector

**Python Implementation:**
```python
import numpy as np

# Creating vectors
vector_a = np.array([1, 2, 3, 4])
vector_b = np.array([5, 6, 7, 8])

# Vector addition
vector_sum = vector_a + vector_b
print(f"Vector sum: {vector_sum}")  # [6, 8, 10, 12]

# Dot product
dot_product = np.dot(vector_a, vector_b)
print(f"Dot product: {dot_product}")  # 70

# Vector magnitude (L2 norm)
magnitude = np.linalg.norm(vector_a)
print(f"Magnitude: {magnitude}")  # 5.477
```

### 3. Matrices

A **matrix** is a 2D array of numbers. In ML, matrices represent:
- Datasets (rows = samples, columns = features)
- Weights in neural networks
- Images (height × width × channels)

![Matrix Representation](https://cdn.kastatic.org/googleusercontent/qZlmAAqYCJXscLFRfPF1hfpopFLiW0aYYa-Uqd7iYPqRSWqlfIgvQrDqMj9VJP61oNG6N4za-kJPzbAwUcOBS_4)

**Mathematical Representation:**
```
    [a₁₁  a₁₂  a₁₃]
A = [a₂₁  a₂₂  a₂₃]
    [a₃₁  a₃₂  a₃₃]
```

**Dimensions**: An m×n matrix has m rows and n columns.

### 4. Tensors

A **tensor** is a generalization of matrices to higher dimensions:
- 0D tensor: Scalar
- 1D tensor: Vector
- 2D tensor: Matrix
- 3D tensor: Cube of numbers (e.g., color image)
- 4D tensor: Batch of images

---

## Matrix Operations

### 1. Matrix Addition and Subtraction

Matrices must have the same dimensions to be added or subtracted.

![Matrix Addition](https://www.mathsisfun.com/algebra/images/matrix-addition.svg)

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Addition
C = A + B  # [[6, 8], [10, 12]]

# Subtraction
D = A - B  # [[-4, -4], [-4, -4]]
```

### 2. Scalar Multiplication

Multiply every element by a scalar:

```python
A = np.array([[1, 2], [3, 4]])
scalar = 3

B = scalar * A  # [[3, 6], [9, 12]]
```

### 3. Matrix Multiplication

The most important operation in ML! For matrices A (m×n) and B (n×p), the product AB is (m×p).

![Matrix Multiplication](https://www.mathsisfun.com/algebra/images/matrix-multiply-a.svg)

**Key Rule**: Number of columns in first matrix = Number of rows in second matrix

```python
A = np.array([[1, 2, 3], 
              [4, 5, 6]])  # 2×3 matrix

B = np.array([[7, 8], 
              [9, 10], 
              [11, 12]])  # 3×2 matrix

C = np.dot(A, B)  # or A @ B in Python 3.5+
# Result: 2×2 matrix
# [[58, 64], [139, 154]]
```

### 4. Element-wise (Hadamard) Product

Multiply corresponding elements (matrices must have same dimensions):

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A * B  # Element-wise
# [[5, 12], [21, 32]]
```

### 5. Matrix Transpose

Flip matrix over its diagonal (rows become columns):

![Matrix Transpose](https://cdn1.byjus.com/wp-content/uploads/2020/10/Transpose-Of-A-Matrix-1.png)

```python
A = np.array([[1, 2, 3], 
              [4, 5, 6]])

A_transpose = A.T
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

**Properties:**
- (Aᵀ)ᵀ = A
- (A + B)ᵀ = Aᵀ + Bᵀ
- (AB)ᵀ = BᵀAᵀ

### 6. Matrix Inverse

For square matrix A, the inverse A⁻¹ satisfies: AA⁻¹ = A⁻¹A = I

```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)

# Verify: A @ A_inv ≈ Identity matrix
identity = A @ A_inv
# [[1, 0], [0, 1]]
```

**Note**: Not all matrices have inverses (must be square and non-singular).

---

## Special Matrices

### 1. Identity Matrix

Square matrix with 1s on diagonal, 0s elsewhere:

```python
I = np.eye(3)
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]
```

**Property**: AI = IA = A

### 2. Diagonal Matrix

Non-zero elements only on the diagonal:

```python
D = np.diag([1, 2, 3])
# [[1, 0, 0],
#  [0, 2, 0],
#  [0, 0, 3]]
```

### 3. Symmetric Matrix

A = Aᵀ (equal to its transpose):

```python
S = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])
# S == S.T
```

### 4. Orthogonal Matrix

Columns are orthonormal (perpendicular unit vectors):
- QᵀQ = QQᵀ = I
- Q⁻¹ = Qᵀ

---

## Linear Transformations

Linear transformations are functions that preserve vector addition and scalar multiplication. In ML, they represent:
- Rotations
- Scaling
- Projections
- Neural network layers

### Geometric Interpretations

![Linear Transformations](https://miro.medium.com/max/1400/1*JKSdr5BmCg8s5EvQ0vrS7g.png)

**1. Rotation Matrix (2D):**
```python
theta = np.pi/4  # 45 degrees
rotation = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

point = np.array([1, 0])
rotated = rotation @ point  # [0.707, 0.707]
```

**2. Scaling Matrix:**
```python
scaling = np.array([[2, 0],    # Scale x by 2
                    [0, 3]])    # Scale y by 3

point = np.array([1, 1])
scaled = scaling @ point  # [2, 3]
```

**3. Projection Matrix:**
Projects vectors onto a subspace (used in PCA):

```python
# Project onto x-axis
projection = np.array([[1, 0],
                       [0, 0]])

point = np.array([3, 4])
projected = projection @ point  # [3, 0]
```

---

## Eigenvalues and Eigenvectors

### Concept

For matrix A, if Av = λv (where v ≠ 0), then:
- λ is an **eigenvalue**
- v is an **eigenvector**

![Eigenvalues Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Eigenvalue_equation.svg/1200px-Eigenvalue_equation.svg.png)

**Interpretation**: The matrix A stretches the eigenvector v by factor λ without changing its direction.

### Calculation

```python
A = np.array([[3, 1],
              [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")  # [4, 2]
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    λ = eigenvalues[i]
    
    Av = A @ v
    λv = λ * v
    
    print(f"Av ≈ λv: {np.allclose(Av, λv)}")  # True
```

### Eigendecomposition

For symmetric matrix A:
A = QΛQᵀ

Where:
- Q: Matrix of eigenvectors
- Λ: Diagonal matrix of eigenvalues

```python
# Eigendecomposition
eigenvalues, Q = np.linalg.eig(A)
Λ = np.diag(eigenvalues)

# Reconstruct A
A_reconstructed = Q @ Λ @ Q.T
print(f"Reconstruction accurate: {np.allclose(A, A_reconstructed)}")
```

---

## Applications in Machine Learning

### 1. Principal Component Analysis (PCA)

PCA uses eigendecomposition to find principal components:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
X = X @ np.array([[2, 1], [1, 2]])  # Correlate features

# Apply PCA
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

# Principal components are eigenvectors of covariance matrix
cov_matrix = np.cov(X.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Principal components:\n{pca.components_}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], alpha=0.5)
ax1.set_title('Original Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# Plot principal components
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    ax1.arrow(0, 0, comp[0]*var*2, comp[1]*var*2, 
              head_width=0.2, head_length=0.2, fc=f'C{i}', ec=f'C{i}')

ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5)
ax2.set_title('PCA Transformed Data')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')

plt.tight_layout()
plt.show()
```

### 2. Linear Regression (Normal Equation)

The optimal weights in linear regression: w = (XᵀX)⁻¹Xᵀy

```python
# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 3)
true_weights = np.array([2, -1, 0.5])
y = X @ true_weights + np.random.randn(100) * 0.1

# Add bias term
X_with_bias = np.c_[np.ones(100), X]

# Normal equation
weights = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y

print(f"True weights: {true_weights}")
print(f"Estimated weights: {weights[1:]}")  # Exclude bias
print(f"Bias term: {weights[0]}")
```

### 3. Singular Value Decomposition (SVD)

SVD decomposes matrix A into: A = UΣVᵀ

Used in:
- Recommendation systems
- Image compression
- Natural Language Processing (LSA)

```python
# Image compression using SVD
from PIL import Image
import requests
from io import BytesIO

# Load a sample image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Einstein_1921_by_F_Schmutzer.jpg/330px-Einstein_1921_by_F_Schmutzer.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
img_array = np.array(img)

# Perform SVD
U, s, Vt = np.linalg.svd(img_array, full_matrices=False)

# Reconstruct with different numbers of components
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
components = [5, 10, 20, 50, 100, 200]

for ax, k in zip(axes.flat, components):
    # Reconstruct using k components
    reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    ax.imshow(reconstructed, cmap='gray')
    ax.set_title(f'{k} components')
    ax.axis('off')
    
    # Calculate compression ratio
    original_size = img_array.shape[0] * img_array.shape[1]
    compressed_size = k * (U.shape[0] + Vt.shape[1] + 1)
    ratio = original_size / compressed_size
    ax.text(0.5, -0.1, f'Compression: {ratio:.1f}x', 
            transform=ax.transAxes, ha='center')

plt.tight_layout()
plt.show()
```

### 4. PageRank Algorithm

Google's PageRank uses eigenvectors to rank web pages:

```python
# Simple PageRank implementation
def pagerank(adjacency_matrix, damping=0.85, max_iter=100):
    n = adjacency_matrix.shape[0]
    
    # Column-normalize adjacency matrix
    M = adjacency_matrix / adjacency_matrix.sum(axis=0, keepdims=True)
    
    # Add damping factor
    M = damping * M + (1 - damping) / n * np.ones((n, n))
    
    # Power iteration to find dominant eigenvector
    v = np.ones(n) / n
    for _ in range(max_iter):
        v = M @ v
        v = v / v.sum()  # Normalize
    
    return v

# Example: Small network
# Page connections: 0->1, 1->2, 2->0, 2->1, 1->3, 3->2
adjacency = np.array([[0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 0, 0, 0]])

ranks = pagerank(adjacency)
for i, rank in enumerate(ranks):
    print(f"Page {i}: {rank:.3f}")
```

---

## Practical Exercises

### Exercise 1: Vector Operations

```python
# Problem: Compute the angle between two vectors
def angle_between_vectors(v1, v2):
    """
    Calculate angle between two vectors in degrees
    """
    # Your code here
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle_rad)

# Test
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
print(f"Angle: {angle_between_vectors(v1, v2)}°")  # Should be 90°
```

### Exercise 2: Matrix Rank

```python
# Problem: Implement a function to check if vectors are linearly independent
def are_linearly_independent(vectors):
    """
    Check if a list of vectors are linearly independent
    """
    matrix = np.array(vectors).T
    rank = np.linalg.matrix_rank(matrix)
    return rank == len(vectors)

# Test cases
vectors1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Independent
vectors2 = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]  # Dependent

print(f"Set 1 independent: {are_linearly_independent(vectors1)}")
print(f"Set 2 independent: {are_linearly_independent(vectors2)}")
```

### Exercise 3: Gram-Schmidt Orthogonalization

```python
def gram_schmidt(vectors):
    """
    Orthogonalize a set of vectors using Gram-Schmidt process
    """
    orthogonal = []
    for v in vectors:
        # Subtract projections onto all previous orthogonal vectors
        for u in orthogonal:
            v = v - np.dot(v, u) / np.dot(u, u) * u
        if np.linalg.norm(v) > 1e-10:  # Check for linear dependence
            orthogonal.append(v / np.linalg.norm(v))  # Normalize
    return np.array(orthogonal)

# Test
vectors = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
orthonormal = gram_schmidt(vectors)

# Verify orthogonality
for i in range(len(orthonormal)):
    for j in range(i+1, len(orthonormal)):
        dot_product = np.dot(orthonormal[i], orthonormal[j])
        print(f"v{i} · v{j} = {dot_product:.10f}")  # Should be ~0
```

### Exercise 4: Power Method for Eigenvalues

```python
def power_method(A, num_iterations=100):
    """
    Find the dominant eigenvalue and eigenvector using power method
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(num_iterations):
        v_new = A @ v
        v_new = v_new / np.linalg.norm(v_new)
        v = v_new
    
    # Compute eigenvalue using Rayleigh quotient
    eigenvalue = v @ A @ v / (v @ v)
    
    return eigenvalue, v

# Test
A = np.array([[3, 1], [1, 3]])
eigenvalue, eigenvector = power_method(A)

print(f"Dominant eigenvalue: {eigenvalue:.4f}")
print(f"Dominant eigenvector: {eigenvector}")

# Compare with numpy
true_eigenvalues, true_eigenvectors = np.linalg.eig(A)
print(f"True eigenvalues: {true_eigenvalues}")
```

---

## Summary and Key Takeaways

### 🎯 Core Concepts Mastered

1. **Data Representation**
   - Scalars, vectors, matrices, and tensors
   - How ML data maps to linear algebra structures

2. **Essential Operations**
   - Matrix multiplication (foundation of neural networks)
   - Transpose (used in backpropagation)
   - Inverse (solving linear systems)

3. **Special Matrices**
   - Identity (neutral element)
   - Diagonal (efficient computations)
   - Symmetric (nice properties for optimization)
   - Orthogonal (preserves distances)

4. **Advanced Concepts**
   - Eigenvalues/eigenvectors (PCA, PageRank)
   - SVD (compression, recommendations)
   - Linear transformations (neural network layers)

### 💡 Key Insights for ML

1. **Everything is Linear Algebra**: From simple linear regression to deep neural networks, it's all matrix operations under the hood.

2. **Computational Efficiency**: Understanding matrix operations helps write efficient code (vectorization vs loops).

3. **Geometric Intuition**: Linear algebra provides geometric interpretation of ML algorithms.

4. **Dimensionality**: Concepts like rank and eigenvalues help understand data dimensionality.


[**Continue to Module 2: Probability and Statistics →**](02_probability_statistics.md)