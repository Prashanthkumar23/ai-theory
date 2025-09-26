# Module 6: Neural Networks and Deep Learning Basics 🧠

[← Previous Module](05_unsupervised_learning.md) | [Back to Main](../README.md) | [Next Module →](07_scikit_learn.md)

## 📋 Table of Contents
1. [Introduction to Neural Networks](#introduction-to-neural-networks)
2. [Perceptron Model](#perceptron-model)
3. [Multi-Layer Perceptrons](#multi-layer-perceptrons)
4. [Backpropagation Algorithm](#backpropagation-algorithm)
5. [Activation Functions](#activation-functions)
6. [Training Neural Networks](#training-neural-networks)
7. [Deep Learning Fundamentals](#deep-learning-fundamentals)
8. [Convolutional Neural Networks (CNN)](#convolutional-neural-networks-cnn)
9. [Practical Implementation](#practical-implementation)
10. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

---

## Introduction to Neural Networks

Neural networks are inspired by the human brain, consisting of interconnected nodes (neurons) that process and transmit information. They excel at learning complex patterns from data.

### Biological Inspiration
```
Biological Neuron          Artificial Neuron
Dendrites       →          Inputs
Cell Body       →          Weighted Sum + Activation
Axon            →          Output
Synapses        →          Weights
```

### Why Neural Networks?
- **Universal Approximation**: Can approximate any continuous function
- **Feature Learning**: Automatically learn relevant features
- **Scalability**: Performance improves with more data
- **Versatility**: Work with various data types (images, text, audio)

---

## Perceptron Model

### The Building Block
The perceptron is the simplest neural network unit, performing binary classification.

### Mathematical Foundation
```
Output = activation(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
       = activation(w^T x + b)
```

### Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert labels to {-1, 1}
        y_ = np.where(y > 0, 1, -1)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Prediction
                y_predicted = np.sign(linear_output)

                # Perceptron update rule
                if y_[idx] * y_predicted <= 0:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

    def score(self, X, y):
        predictions = self.predict(X)
        y_ = np.where(y > 0, 1, -1)
        accuracy = np.mean(predictions == y_)
        return accuracy

# Example: XOR problem (perceptron will fail)
def demonstrate_perceptron_limitations():
    # Linearly separable data (AND gate)
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])

    # Non-linearly separable data (XOR gate)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train on AND gate
    perceptron_and = Perceptron()
    perceptron_and.fit(X_and, y_and)

    axes[0].scatter(X_and[y_and == 0][:, 0], X_and[y_and == 0][:, 1],
                   color='blue', marker='o', label='Class 0')
    axes[0].scatter(X_and[y_and == 1][:, 0], X_and[y_and == 1][:, 1],
                   color='red', marker='s', label='Class 1')
    axes[0].set_title(f'AND Gate (Accuracy: {perceptron_and.score(X_and, y_and):.2f})')
    axes[0].legend()

    # Try XOR gate
    perceptron_xor = Perceptron()
    perceptron_xor.fit(X_xor, y_xor)

    axes[1].scatter(X_xor[y_xor == 0][:, 0], X_xor[y_xor == 0][:, 1],
                   color='blue', marker='o', label='Class 0')
    axes[1].scatter(X_xor[y_xor == 1][:, 0], X_xor[y_xor == 1][:, 1],
                   color='red', marker='s', label='Class 1')
    axes[1].set_title(f'XOR Gate (Accuracy: {perceptron_xor.score(X_xor, y_xor):.2f})')
    axes[1].legend()

    plt.suptitle('Perceptron Limitations: Linear Separability')
    plt.show()

demonstrate_perceptron_limitations()
```

---

## Multi-Layer Perceptrons

### Architecture
MLPs have multiple layers of neurons, enabling them to learn non-linear patterns.

```python
class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.001):
        """
        layer_sizes: list of layer dimensions [input_size, hidden1, hidden2, ..., output_size]
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU
            if activation == 'relu':
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            # Xavier initialization for sigmoid/tanh
            else:
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1 / layer_sizes[i])

            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def activation_function(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)

    def activation_derivative(self, z):
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self.activation_function(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2

    def forward_propagation(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.z_values = []

        current_input = X

        for i in range(len(self.weights) - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.activation_function(z)
            self.activations.append(a)
            current_input = a

        # Output layer (linear for regression, sigmoid for binary classification)
        z = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid for output
        self.activations.append(output)

        return output

    def backward_propagation(self, X, y):
        """Backpropagation to compute gradients"""
        m = X.shape[0]
        gradients_w = []
        gradients_b = []

        # Output layer gradient
        dz = self.activations[-1] - y.reshape(-1, 1)

        # Iterate backwards through layers
        for i in range(len(self.weights) - 1, -1, -1):
            dw = (1/m) * np.dot(self.activations[i].T, dz)
            db = (1/m) * np.sum(dz, axis=0, keepdims=True)

            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)

            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                dz = da * self.activation_derivative(self.z_values[i-1])

        return gradients_w, gradients_b

    def update_parameters(self, gradients_w, gradients_b):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def fit(self, X, y, epochs=1000, verbose=True):
        """Train the neural network"""
        losses = []

        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)

            # Compute loss (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)

            # Backward propagation
            gradients_w, gradients_b = self.backward_propagation(X, y)

            # Update parameters
            self.update_parameters(gradients_w, gradients_b)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        output = self.forward_propagation(X)
        return (output > 0.5).astype(int)

# Solve XOR with MLP
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Create network with hidden layer
nn = NeuralNetwork([2, 4, 1], activation='relu', learning_rate=0.1)
losses = nn.fit(X_xor, y_xor, epochs=1000, verbose=False)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss curve
ax1.plot(losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True)

# Decision boundary
h = 0.01
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax2.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
ax2.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, cmap='RdBu', edgecolor='black', s=100)
ax2.set_title('XOR Solution with MLP')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')

plt.tight_layout()
plt.show()

print(f"Final predictions: {nn.predict(X_xor).flatten()}")
print(f"True labels: {y_xor}")
```

---

## Backpropagation Algorithm

### The Chain Rule
Backpropagation efficiently computes gradients using the chain rule of calculus.

### Visual Understanding

```python
def visualize_backpropagation():
    """Visualize gradient flow in backpropagation"""

    # Create a simple 2-layer network visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    # Network structure
    layers = [3, 4, 2]  # 3 inputs, 4 hidden, 2 outputs
    layer_positions = [0.2, 0.5, 0.8]

    # Draw neurons
    neurons = {}
    for layer_idx, (n_neurons, x_pos) in enumerate(zip(layers, layer_positions)):
        for neuron_idx in range(n_neurons):
            y_pos = (neuron_idx + 1) / (n_neurons + 1)
            circle = plt.Circle((x_pos, y_pos), 0.03, color='lightblue', ec='black')
            ax.add_patch(circle)
            neurons[(layer_idx, neuron_idx)] = (x_pos, y_pos)

    # Draw connections with gradient flow
    for layer_idx in range(len(layers) - 1):
        for i in range(layers[layer_idx]):
            for j in range(layers[layer_idx + 1]):
                start = neurons[(layer_idx, i)]
                end = neurons[(layer_idx + 1, j)]

                # Forward pass (blue)
                ax.arrow(start[0], start[1], end[0] - start[0] - 0.03, end[1] - start[1],
                        head_width=0.01, head_length=0.01, fc='blue', alpha=0.3)

                # Backward pass (red)
                ax.arrow(end[0] - 0.01, end[1], -(end[0] - start[0] - 0.04), -(end[1] - start[1]),
                        head_width=0.01, head_length=0.01, fc='red', alpha=0.3)

    # Labels
    ax.text(0.2, -0.05, 'Input Layer', ha='center', fontsize=12)
    ax.text(0.5, -0.05, 'Hidden Layer', ha='center', fontsize=12)
    ax.text(0.8, -0.05, 'Output Layer', ha='center', fontsize=12)

    ax.text(0.5, 1.05, 'Forward Pass →', color='blue', ha='center', fontsize=12)
    ax.text(0.5, 1.00, '← Backward Pass', color='red', ha='center', fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Backpropagation: Gradient Flow', fontsize=14, fontweight='bold')

    plt.show()

visualize_backpropagation()
```

### Detailed Backpropagation Implementation

```python
class DetailedBackprop:
    """Step-by-step backpropagation with detailed calculations"""

    def __init__(self):
        # Simple 2-2-1 network
        np.random.seed(42)
        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward_and_backward(self, X, y):
        """Detailed forward and backward pass"""
        print("=" * 50)
        print("FORWARD PASS")
        print("=" * 50)

        # Input
        print(f"Input X:\n{X}")

        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1
        print(f"\nz1 = X @ W1 + b1:\n{z1}")

        a1 = self.sigmoid(z1)
        print(f"\na1 = sigmoid(z1):\n{a1}")

        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        print(f"\nz2 = a1 @ W2 + b2:\n{z2}")

        a2 = self.sigmoid(z2)
        print(f"\na2 = sigmoid(z2) [Output]:\n{a2}")

        # Loss
        loss = 0.5 * (a2 - y) ** 2
        print(f"\nLoss = 0.5 * (a2 - y)²:\n{loss}")

        print("\n" + "=" * 50)
        print("BACKWARD PASS")
        print("=" * 50)

        # Output layer gradients
        dL_da2 = a2 - y
        print(f"∂L/∂a2 = a2 - y:\n{dL_da2}")

        da2_dz2 = self.sigmoid_derivative(z2)
        print(f"\n∂a2/∂z2 = σ'(z2):\n{da2_dz2}")

        dL_dz2 = dL_da2 * da2_dz2
        print(f"\n∂L/∂z2 = ∂L/∂a2 * ∂a2/∂z2:\n{dL_dz2}")

        dL_dW2 = np.dot(a1.T, dL_dz2)
        print(f"\n∂L/∂W2 = a1.T @ ∂L/∂z2:\n{dL_dW2}")

        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        print(f"\n∂L/∂b2 = sum(∂L/∂z2):\n{dL_db2}")

        # Hidden layer gradients
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        print(f"\n∂L/∂a1 = ∂L/∂z2 @ W2.T:\n{dL_da1}")

        da1_dz1 = self.sigmoid_derivative(z1)
        print(f"\n∂a1/∂z1 = σ'(z1):\n{da1_dz1}")

        dL_dz1 = dL_da1 * da1_dz1
        print(f"\n∂L/∂z1 = ∂L/∂a1 * ∂a1/∂z1:\n{dL_dz1}")

        dL_dW1 = np.dot(X.T, dL_dz1)
        print(f"\n∂L/∂W1 = X.T @ ∂L/∂z1:\n{dL_dW1}")

        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        print(f"\n∂L/∂b1 = sum(∂L/∂z1):\n{dL_db1}")

        return dL_dW1, dL_db1, dL_dW2, dL_db2

# Example with single data point
backprop_demo = DetailedBackprop()
X_single = np.array([[1, 0]])
y_single = np.array([[1]])
gradients = backprop_demo.forward_and_backward(X_single, y_single)
```

---

## Activation Functions

### Common Activation Functions

```python
def plot_activation_functions():
    """Visualize common activation functions and their derivatives"""

    x = np.linspace(-5, 5, 100)

    # Define activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(x):
        return np.tanh(x)

    def relu(x):
        return np.maximum(0, x)

    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def swish(x):
        return x * sigmoid(x)

    # Define derivatives
    def sigmoid_prime(x):
        s = sigmoid(x)
        return s * (1 - s)

    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2

    def relu_prime(x):
        return (x > 0).astype(float)

    def leaky_relu_prime(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    activations = [
        ('Sigmoid', sigmoid, sigmoid_prime),
        ('Tanh', tanh, tanh_prime),
        ('ReLU', relu, relu_prime),
        ('Leaky ReLU', leaky_relu, leaky_relu_prime),
    ]

    for idx, (name, func, deriv) in enumerate(activations):
        row = idx // 2
        col = (idx % 2) * 2

        # Function
        axes[row, col].plot(x, func(x), 'b-', linewidth=2)
        axes[row, col].set_title(f'{name}')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_xlabel('x')
        axes[row, col].set_ylabel('f(x)')
        axes[row, col].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[row, col].axvline(x=0, color='k', linestyle='-', alpha=0.3)

        # Derivative
        axes[row, col+1].plot(x, deriv(x), 'r-', linewidth=2)
        axes[row, col+1].set_title(f'{name} Derivative')
        axes[row, col+1].grid(True, alpha=0.3)
        axes[row, col+1].set_xlabel('x')
        axes[row, col+1].set_ylabel("f'(x)")
        axes[row, col+1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[row, col+1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Additional activations
    axes[2, 0].plot(x, elu(x), 'b-', linewidth=2)
    axes[2, 0].set_title('ELU')
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(x, swish(x), 'b-', linewidth=2)
    axes[2, 1].set_title('Swish')
    axes[2, 1].grid(True, alpha=0.3)

    # Comparison
    axes[2, 2].plot(x, sigmoid(x), label='Sigmoid', alpha=0.7)
    axes[2, 2].plot(x, tanh(x), label='Tanh', alpha=0.7)
    axes[2, 2].plot(x, relu(x), label='ReLU', alpha=0.7)
    axes[2, 2].set_title('Comparison')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

    # Vanishing gradient visualization
    z = np.linspace(-10, 10, 100)
    grad_sigmoid = sigmoid_prime(z)
    axes[2, 3].semilogy(z, grad_sigmoid + 1e-10, label='Sigmoid')
    axes[2, 3].semilogy(z, tanh_prime(z) + 1e-10, label='Tanh')
    axes[2, 3].semilogy(z, relu_prime(z) + 1e-10, label='ReLU')
    axes[2, 3].set_title('Gradient Magnitude (log scale)')
    axes[2, 3].legend()
    axes[2, 3].grid(True, alpha=0.3)

    plt.suptitle('Activation Functions and Their Derivatives', fontsize=16)
    plt.tight_layout()
    plt.show()

plot_activation_functions()
```

### Softmax for Multi-class Classification

```python
def softmax(z):
    """Stable softmax implementation"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """Cross-entropy loss for multi-class classification"""
    n_samples = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(n_samples), y_true] + 1e-8)
    return np.mean(log_likelihood)

# Example: Multi-class classification
def multiclass_demo():
    # Generate sample data
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import OneHotEncoder

    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                              n_redundant=0, n_clusters_per_class=1, n_classes=3)

    # One-hot encode labels
    y_onehot = np.eye(3)[y]

    # Visualize
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Multi-class Classification Problem')
    plt.show()

    # Apply softmax to random logits
    logits = np.random.randn(5, 3)  # 5 samples, 3 classes
    probabilities = softmax(logits)

    print("Logits:")
    print(logits)
    print("\nSoftmax probabilities:")
    print(probabilities)
    print("\nSum of probabilities per sample:", probabilities.sum(axis=1))

multiclass_demo()
```

---

## Training Neural Networks

### Weight Initialization Strategies

```python
def compare_weight_initialization():
    """Compare different weight initialization strategies"""

    n_input = 100
    n_output = 50
    n_samples = 1000

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Different initialization methods
    initializations = {
        'Zero': np.zeros((n_input, n_output)),
        'Random Normal': np.random.randn(n_input, n_output),
        'Random Uniform': np.random.uniform(-1, 1, (n_input, n_output)),
        'Xavier/Glorot': np.random.randn(n_input, n_output) * np.sqrt(1 / n_input),
        'He (ReLU)': np.random.randn(n_input, n_output) * np.sqrt(2 / n_input),
        'LeCun': np.random.randn(n_input, n_output) * np.sqrt(1 / n_input)
    }

    for idx, (name, weights) in enumerate(initializations.items()):
        ax = axes[idx // 3, idx % 3]

        # Simulate forward pass
        X = np.random.randn(n_samples, n_input)
        Z = np.dot(X, weights)

        # Plot histogram of activations
        ax.hist(Z.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(f'{name}\nμ={np.mean(Z):.3f}, σ={np.std(Z):.3f}')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    plt.suptitle('Weight Initialization Effects on Activations', fontsize=14)
    plt.tight_layout()
    plt.show()

compare_weight_initialization()
```

### Batch Normalization

```python
class BatchNorm:
    """Batch Normalization layer implementation"""

    def __init__(self, n_features, momentum=0.9, eps=1e-5):
        self.n_features = n_features
        self.momentum = momentum
        self.eps = eps

        # Parameters to be learned
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)

        # Running statistics for inference
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)

        self.training = True

    def forward(self, X):
        if self.training:
            # Calculate batch statistics
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)

            # Normalize
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.eps)

            # Scale and shift
            output = self.gamma * X_norm + self.beta

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            # Store for backward pass
            self.X_norm = X_norm
            self.batch_mean = batch_mean
            self.batch_var = batch_var

        else:
            # Use running statistics during inference
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
            output = self.gamma * X_norm + self.beta

        return output

    def backward(self, dout):
        # Gradient computation for backpropagation
        N = dout.shape[0]

        # Gradients w.r.t. scale and shift
        dgamma = np.sum(dout * self.X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        # Gradient w.r.t. normalized input
        dX_norm = dout * self.gamma

        # Gradient w.r.t. variance
        dvar = np.sum(dX_norm * (self.X - self.batch_mean), axis=0) * \
               (-0.5) * (self.batch_var + self.eps) ** (-1.5)

        # Gradient w.r.t. mean
        dmean = np.sum(dX_norm * (-1 / np.sqrt(self.batch_var + self.eps)), axis=0) + \
                dvar * np.mean(-2 * (self.X - self.batch_mean), axis=0)

        # Gradient w.r.t. input
        dX = dX_norm / np.sqrt(self.batch_var + self.eps) + \
             dvar * 2 * (self.X - self.batch_mean) / N + \
             dmean / N

        return dX

# Demonstrate batch normalization effect
def demonstrate_batchnorm():
    # Generate data with covariate shift
    X_train = np.random.randn(1000, 10) * 3 + 2
    X_test = np.random.randn(100, 10) * 5 - 1  # Different distribution

    bn = BatchNorm(10)

    # Apply batch norm
    X_train_bn = bn.forward(X_train)
    bn.training = False
    X_test_bn = bn.forward(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Before batch norm
    axes[0].hist(X_train.flatten(), bins=50, alpha=0.5, label='Train', color='blue')
    axes[0].hist(X_test.flatten(), bins=50, alpha=0.5, label='Test', color='red')
    axes[0].set_title('Before Batch Normalization')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # After batch norm
    axes[1].hist(X_train_bn.flatten(), bins=50, alpha=0.5, label='Train', color='blue')
    axes[1].hist(X_test_bn.flatten(), bins=50, alpha=0.5, label='Test', color='red')
    axes[1].set_title('After Batch Normalization')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

demonstrate_batchnorm()
```

### Dropout Regularization

```python
class Dropout:
    """Dropout layer for regularization"""

    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.training = True
        self.mask = None

    def forward(self, X):
        if self.training:
            # Create dropout mask
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, X.shape)
            # Apply mask and scale
            return X * self.mask / (1 - self.dropout_rate)
        else:
            # No dropout during inference
            return X

    def backward(self, dout):
        if self.training:
            return dout * self.mask / (1 - self.dropout_rate)
        return dout

# Visualize dropout effect
def visualize_dropout():
    # Create sample hidden layer activations
    n_neurons = 100
    activations = np.random.randn(1, n_neurons)

    dropout_rates = [0.0, 0.2, 0.5, 0.8]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, rate in enumerate(dropout_rates):
        dropout = Dropout(dropout_rate=rate)
        output = dropout.forward(activations.copy())

        axes[idx].bar(range(n_neurons), output.flatten(), color='blue', alpha=0.6)
        axes[idx].set_title(f'Dropout Rate = {rate}')
        axes[idx].set_xlabel('Neuron Index')
        axes[idx].set_ylabel('Activation')
        axes[idx].set_ylim([-3, 3])

        # Mark dropped neurons
        if rate > 0:
            dropped = np.where(dropout.mask[0] == 0)[0]
            for d in dropped:
                axes[idx].axvspan(d-0.4, d+0.4, color='red', alpha=0.3)

    plt.suptitle('Effect of Dropout on Activations', fontsize=14)
    plt.tight_layout()
    plt.show()

visualize_dropout()
```

### Optimization Algorithms

```python
class Optimizers:
    """Different optimization algorithms"""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def sgd(self, weights, gradients):
        """Vanilla SGD"""
        return weights - self.learning_rate * gradients

    def momentum(self, weights, gradients, velocity, beta=0.9):
        """SGD with momentum"""
        velocity = beta * velocity - self.learning_rate * gradients
        return weights + velocity, velocity

    def rmsprop(self, weights, gradients, cache, beta=0.999, eps=1e-8):
        """RMSprop optimizer"""
        cache = beta * cache + (1 - beta) * gradients ** 2
        return weights - self.learning_rate * gradients / (np.sqrt(cache) + eps), cache

    def adam(self, weights, gradients, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer"""
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * gradients ** 2

        # Bias correction
        m_corrected = m / (1 - beta1 ** t)
        v_corrected = v / (1 - beta2 ** t)

        weights = weights - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + eps)
        return weights, m, v

def compare_optimizers():
    """Compare convergence of different optimizers"""

    # Create a simple optimization landscape
    def loss_function(x, y):
        return 0.1 * x**2 + 2 * y**2

    def gradient(x, y):
        return 0.2 * x, 4 * y

    # Starting point
    x_start, y_start = 8.0, 3.0

    # Track paths for different optimizers
    paths = {
        'SGD': [[x_start, y_start]],
        'Momentum': [[x_start, y_start]],
        'RMSprop': [[x_start, y_start]],
        'Adam': [[x_start, y_start]]
    }

    # Initialize optimizer states
    velocity = np.array([0.0, 0.0])
    cache = np.array([0.0, 0.0])
    m = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])

    opt = Optimizers(learning_rate=0.1)

    # Run optimization
    for t in range(1, 51):
        for name in paths.keys():
            x, y = paths[name][-1]
            grad_x, grad_y = gradient(x, y)
            grad = np.array([grad_x, grad_y])

            if name == 'SGD':
                new_pos = opt.sgd(np.array([x, y]), grad)
            elif name == 'Momentum':
                new_pos, velocity = opt.momentum(np.array([x, y]), grad, velocity)
            elif name == 'RMSprop':
                new_pos, cache = opt.rmsprop(np.array([x, y]), grad, cache)
            elif name == 'Adam':
                new_pos, m, v = opt.adam(np.array([x, y]), grad, m, v, t)

            paths[name].append(new_pos.tolist())

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create contour plot
    x = np.linspace(-1, 10, 100)
    y = np.linspace(-1, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_function(X, Y)

    contour = ax.contour(X, Y, Z, levels=20, alpha=0.4)
    ax.clabel(contour, inline=True, fontsize=8)

    # Plot optimization paths
    colors = {'SGD': 'blue', 'Momentum': 'red', 'RMSprop': 'green', 'Adam': 'purple'}
    for name, path in paths.items():
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'o-', color=colors[name], label=name, markersize=4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Optimizer Comparison on Simple Loss Landscape')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()

compare_optimizers()
```

---

## Deep Learning Fundamentals

### Why Deep Networks?

```python
def demonstrate_deep_vs_shallow():
    """Show the power of depth in neural networks"""

    from sklearn.datasets import make_circles
    from sklearn.model_selection import train_test_split

    # Generate non-linearly separable data
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train shallow network (1 hidden layer)
    shallow = NeuralNetwork([2, 100, 1], activation='relu', learning_rate=0.01)
    shallow_losses = shallow.fit(X_train, y_train, epochs=500, verbose=False)

    # Train deep network (3 hidden layers)
    deep = NeuralNetwork([2, 20, 20, 20, 1], activation='relu', learning_rate=0.01)
    deep_losses = deep.fit(X_train, y_train, epochs=500, verbose=False)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Data
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', alpha=0.6)
    axes[0, 0].set_title('Original Data')

    # Training curves
    axes[0, 1].plot(shallow_losses, label='Shallow (100 hidden)', alpha=0.7)
    axes[0, 1].plot(deep_losses, label='Deep (3x20 hidden)', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Decision boundaries
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Shallow network decision boundary
    Z_shallow = shallow.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_shallow = Z_shallow.reshape(xx.shape)
    axes[1, 0].contourf(xx, yy, Z_shallow, alpha=0.4, cmap='RdBu')
    axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdBu', edgecolor='black')
    axes[1, 0].set_title('Shallow Network Decision Boundary')

    # Deep network decision boundary
    Z_deep = deep.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_deep = Z_deep.reshape(xx.shape)
    axes[1, 1].contourf(xx, yy, Z_deep, alpha=0.4, cmap='RdBu')
    axes[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdBu', edgecolor='black')
    axes[1, 1].set_title('Deep Network Decision Boundary')

    plt.suptitle('Deep vs Shallow Networks', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Print accuracies
    shallow_acc = np.mean(shallow.predict(X_test) == y_test.reshape(-1, 1))
    deep_acc = np.mean(deep.predict(X_test) == y_test.reshape(-1, 1))
    print(f"Shallow Network Accuracy: {shallow_acc:.3f}")
    print(f"Deep Network Accuracy: {deep_acc:.3f}")

demonstrate_deep_vs_shallow()
```

---

## Convolutional Neural Networks (CNN)

### Basic Convolution Operation

```python
def demonstrate_convolution():
    """Visualize 2D convolution operation"""

    # Create sample image
    image = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [2, 1, 0, 1, 2],
        [1, 0, 1, 2, 3]
    ])

    # Define kernels
    kernels = {
        'Edge Horizontal': np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]]),
        'Edge Vertical': np.array([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]]),
        'Sharpen': np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]]),
        'Blur': np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]]) / 9
    }

    def conv2d(image, kernel):
        """Simple 2D convolution"""
        h, w = image.shape
        kh, kw = kernel.shape
        output = np.zeros((h - kh + 1, w - kw + 1))

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)

        return output

    # Apply convolutions
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Apply different kernels
    positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
    for (name, kernel), (i, j) in zip(kernels.items(), positions):
        output = conv2d(image, kernel)
        axes[i, j].imshow(output, cmap='gray')
        axes[i, j].set_title(name)
        axes[i, j].axis('off')

    # Visualize kernel
    axes[1, 2].imshow(kernels['Edge Horizontal'], cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 2].set_title('Edge Horizontal Kernel')
    for i in range(3):
        for j in range(3):
            axes[1, 2].text(j, i, f'{kernels["Edge Horizontal"][i, j]:.1f}',
                          ha='center', va='center', color='white' if abs(kernels["Edge Horizontal"][i, j]) > 0.5 else 'black')
    axes[1, 2].axis('off')

    plt.suptitle('Convolution Operations', fontsize=14)
    plt.tight_layout()
    plt.show()

demonstrate_convolution()
```

### Simple CNN Implementation

```python
class SimpleCNN:
    """Basic CNN for image classification"""

    def __init__(self):
        # Conv layer parameters
        self.conv_filters = np.random.randn(8, 3, 3) * 0.1  # 8 filters of 3x3

        # Fully connected layer
        self.fc_weights = None
        self.fc_bias = None

    def conv2d(self, image, kernel):
        """2D convolution"""
        h, w = image.shape
        kh, kw = kernel.shape
        output = np.zeros((h - kh + 1, w - kw + 1))

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)

        return output

    def relu(self, x):
        return np.maximum(0, x)

    def maxpool2d(self, x, pool_size=2):
        """2D max pooling"""
        h, w = x.shape
        h_out = h // pool_size
        w_out = w // pool_size
        output = np.zeros((h_out, w_out))

        for i in range(h_out):
            for j in range(w_out):
                output[i, j] = np.max(x[i*pool_size:(i+1)*pool_size,
                                       j*pool_size:(j+1)*pool_size])

        return output

    def forward(self, image):
        """Forward pass through CNN"""
        # Convolution layer
        conv_outputs = []
        for kernel in self.conv_filters:
            conv = self.conv2d(image, kernel)
            conv = self.relu(conv)
            conv = self.maxpool2d(conv)
            conv_outputs.append(conv)

        # Flatten
        features = np.concatenate([c.flatten() for c in conv_outputs])

        # Initialize FC layer if needed
        if self.fc_weights is None:
            self.fc_weights = np.random.randn(len(features), 10) * 0.1
            self.fc_bias = np.zeros(10)

        # Fully connected layer
        output = np.dot(features, self.fc_weights) + self.fc_bias

        return output, conv_outputs

    def visualize_features(self, image):
        """Visualize feature maps"""
        output, conv_outputs = self.forward(image)

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))

        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        # Feature maps
        for idx in range(min(8, len(conv_outputs))):
            i = (idx + 1) // 3
            j = (idx + 1) % 3
            axes[i, j].imshow(conv_outputs[idx], cmap='gray')
            axes[i, j].set_title(f'Feature Map {idx+1}')
            axes[i, j].axis('off')

        plt.suptitle('CNN Feature Maps', fontsize=14)
        plt.tight_layout()
        plt.show()

# Create and visualize CNN
cnn = SimpleCNN()
sample_image = np.random.randn(28, 28)
cnn.visualize_features(sample_image)
```

---

## Practical Implementation

### Using TensorFlow/Keras

```python
# Note: This is example code. Install TensorFlow to run: pip install tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_keras_model():
    """Build a neural network using Keras"""

    # Sequential model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Functional API example
def build_functional_model():
    """Build model using Keras Functional API"""

    inputs = keras.Input(shape=(784,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# CNN example
def build_cnn():
    """Build a CNN for image classification"""

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return model

print("Model architectures defined (requires TensorFlow to instantiate)")
```

### PyTorch Example

```python
# Note: This is example code. Install PyTorch to run: pip install torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchNet(nn.Module):
    """Neural network using PyTorch"""

    def __init__(self, input_size, hidden_size, num_classes):
        super(PyTorchNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

class PyTorchCNN(nn.Module):
    """CNN using PyTorch"""

    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

print("PyTorch models defined (requires PyTorch to instantiate)")
```

---

## Common Pitfalls & Solutions

### 1. Vanishing/Exploding Gradients

```python
def demonstrate_gradient_problems():
    """Show vanishing and exploding gradient problems"""

    # Deep network with poor initialization
    n_layers = 20
    layer_size = 100

    # Case 1: Small initialization (vanishing gradients)
    weights_small = [np.random.randn(layer_size, layer_size) * 0.01 for _ in range(n_layers)]

    # Case 2: Large initialization (exploding gradients)
    weights_large = [np.random.randn(layer_size, layer_size) * 1.0 for _ in range(n_layers)]

    # Case 3: Good initialization (He)
    weights_good = [np.random.randn(layer_size, layer_size) * np.sqrt(2/layer_size) for _ in range(n_layers)]

    def forward_pass(x, weights, activation='tanh'):
        activations = [x]
        for w in weights:
            z = np.dot(activations[-1], w)
            if activation == 'tanh':
                a = np.tanh(z)
            elif activation == 'relu':
                a = np.maximum(0, z)
            activations.append(a)
        return activations

    # Test forward pass
    x = np.random.randn(1, layer_size)

    activations_small = forward_pass(x, weights_small)
    activations_large = forward_pass(x, weights_large)
    activations_good = forward_pass(x, weights_good)

    # Plot activation magnitudes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot([np.std(a) for a in activations_small])
    axes[0].set_title('Vanishing Activations\n(Small Init)')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Std Dev of Activations')
    axes[0].set_yscale('log')

    axes[1].plot([np.std(a) for a in activations_large])
    axes[1].set_title('Exploding Activations\n(Large Init)')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Std Dev of Activations')
    axes[1].set_yscale('log')

    axes[2].plot([np.std(a) for a in activations_good])
    axes[2].set_title('Stable Activations\n(He Init)')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Std Dev of Activations')
    axes[2].set_yscale('log')

    plt.suptitle('Gradient Flow Problems and Solutions', fontsize=14)
    plt.tight_layout()
    plt.show()

demonstrate_gradient_problems()
```

### 2. Overfitting

```python
def demonstrate_overfitting():
    """Show overfitting and regularization effects"""

    # Generate synthetic data
    np.random.seed(42)
    n_train = 100
    n_test = 50

    X_train = np.random.randn(n_train, 20)
    y_train = (np.sum(X_train[:, :5], axis=1) > 0).astype(int)

    X_test = np.random.randn(n_test, 20)
    y_test = (np.sum(X_test[:, :5], axis=1) > 0).astype(int)

    # Train models with different regularization
    models = {
        'No Regularization': NeuralNetwork([20, 100, 100, 1], learning_rate=0.01),
        'With Dropout': NeuralNetwork([20, 50, 50, 1], learning_rate=0.01),  # Simulated
        'Small Network': NeuralNetwork([20, 10, 1], learning_rate=0.01)
    }

    results = {}
    for name, model in models.items():
        losses = model.fit(X_train, y_train, epochs=500, verbose=False)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = np.mean(train_pred.flatten() == y_train)
        test_acc = np.mean(test_pred.flatten() == y_test)

        results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'losses': losses
        }

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training curves
    for name, result in results.items():
        ax1.plot(result['losses'], label=name, alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy comparison
    names = list(results.keys())
    train_accs = [results[n]['train_acc'] for n in names]
    test_accs = [results[n]['test_acc'] for n in names]

    x_pos = np.arange(len(names))
    width = 0.35

    ax2.bar(x_pos - width/2, train_accs, width, label='Train', alpha=0.7)
    ax2.bar(x_pos + width/2, test_accs, width, label='Test', alpha=0.7)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Train vs Test Accuracy')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Overfitting and Regularization', fontsize=14)
    plt.tight_layout()
    plt.show()

demonstrate_overfitting()
```

### 3. Debugging Neural Networks

```python
def neural_network_debugging_guide():
    """Common debugging strategies for neural networks"""

    debugging_checklist = """
    Neural Network Debugging Checklist:

    1. DATA CHECKS
    □ Check data shape and types
    □ Look for NaN/Inf values
    □ Verify label encoding
    □ Check train/test split
    □ Normalize/standardize features

    2. ARCHITECTURE CHECKS
    □ Verify input/output dimensions
    □ Check activation functions
    □ Ensure proper layer connections
    □ Validate loss function choice

    3. TRAINING CHECKS
    □ Monitor loss (should decrease)
    □ Check gradient magnitudes
    □ Verify learning rate
    □ Look for overfitting
    □ Test with small dataset first

    4. COMMON FIXES
    • Loss not decreasing → Lower learning rate
    • Loss is NaN → Check for numerical instability
    • Overfitting → Add regularization
    • Underfitting → Increase capacity
    • Slow training → Check batch size
    • Poor generalization → More data or augmentation
    """

    print(debugging_checklist)

    # Example: Gradient checking
    def gradient_check(model, X, y, epsilon=1e-7):
        """Numerical gradient checking"""
        # Implementation of numerical gradient checking
        print("Performing gradient check...")
        # This would compare analytical gradients with numerical approximations
        pass

neural_network_debugging_guide()
```

---

## Practice Exercises

### Exercise 1: Implement Backpropagation
```python
# TODO: Implement full backpropagation for a 3-layer network
# Include forward pass, loss calculation, and gradient computation
```

### Exercise 2: Build Custom Activation
```python
# TODO: Implement Swish, GELU, or Mish activation function
# Compare with standard activations on a classification task
```

### Exercise 3: Implement Batch Normalization
```python
# TODO: Add batch normalization to the neural network class
# Compare training with and without batch norm
```

### Exercise 4: Create a Mini Deep Learning Framework
```python
# TODO: Build a modular framework with:
# - Layer abstraction
# - Multiple optimizers
# - Automatic differentiation
# - Model serialization
```

---

## Summary and Key Takeaways

### Neural Network Essentials
1. **Perceptron**: Linear classifier, building block
2. **MLP**: Multiple layers enable non-linear learning
3. **Backpropagation**: Efficient gradient computation
4. **Activation Functions**: Non-linearity is crucial
5. **Optimization**: Various algorithms for different scenarios
6. **Regularization**: Prevent overfitting
7. **Deep Learning**: Depth enables hierarchical features

### Best Practices Checklist
- [ ] Normalize input data
- [ ] Use appropriate weight initialization
- [ ] Choose suitable activation functions
- [ ] Monitor training and validation loss
- [ ] Implement early stopping
- [ ] Use batch normalization for deep networks
- [ ] Apply dropout for regularization
- [ ] Experiment with learning rates
- [ ] Check gradients when debugging
- [ ] Start simple, then increase complexity

### Architecture Guidelines

| Problem Type | Output Layer | Loss Function |
|-------------|--------------|---------------|
| Binary Classification | Sigmoid | Binary Cross-entropy |
| Multi-class Classification | Softmax | Categorical Cross-entropy |
| Regression | Linear | MSE or MAE |
| Multi-label | Sigmoid | Binary Cross-entropy (per label) |

### When to Use Neural Networks?

**Good for:**
- Complex non-linear patterns
- Large amounts of data
- Image, text, audio processing
- Feature learning needed
- End-to-end learning

**Not ideal for:**
- Small datasets
- Need for interpretability
- Linear relationships
- Tabular data (try trees first)
- Limited computational resources

---

## Additional Resources

### Frameworks to Learn
- **TensorFlow/Keras**: High-level, production-ready
- **PyTorch**: Research-friendly, dynamic graphs
- **JAX**: Functional, high-performance
- **FastAI**: Built on PyTorch, beginner-friendly

### Recommended Projects
1. Build MNIST digit classifier from scratch
2. Implement autoencoder for dimensionality reduction
3. Create sentiment analysis model
4. Build image classifier with data augmentation
5. Implement GAN for image generation

### Next Steps
- Module 7: Master scikit-learn
- Advanced: CNNs for computer vision
- Advanced: RNNs for sequences
- Advanced: Transformers for NLP
- Advanced: Reinforcement learning

---

[**Continue to Module 7: Scikit-learn Introduction →**](07_scikit_learn.md)