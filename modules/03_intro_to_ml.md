# Module 3: Introduction to Machine Learning 🧠

[← Previous Module](02_probability_statistics.md) | [Back to Main](../README.md) | [Next Module →](04_supervised_learning.md)

## 📋 Table of Contents
1. [What is Machine Learning?](#what-is-machine-learning)
2. [Types of Machine Learning](#types-of-machine-learning)
3. [The Machine Learning Pipeline](#the-machine-learning-pipeline)
4. [Training, Validation, and Test Sets](#training-validation-and-test-sets)
5. [Overfitting and Underfitting](#overfitting-and-underfitting)
6. [Bias-Variance Tradeoff](#bias-variance-tradeoff)
7. [Model Evaluation Metrics](#model-evaluation-metrics)
8. [Cross-Validation](#cross-validation)
9. [Feature Engineering](#feature-engineering)
10. [Practical Examples](#practical-examples)
11. [Summary and Key Takeaways](#summary-and-key-takeaways)

---

## What is Machine Learning?

Machine Learning is the science of getting computers to learn and act like humans do, improving their learning over time in an autonomous fashion, by feeding them data and information in the form of observations and real-world interactions.

![ML Overview](https://miro.medium.com/max/1400/1*c_fiB-YgbnMl6nntYGBMHQ.jpeg)

### Traditional Programming vs Machine Learning

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Traditional Programming Example
def traditional_spam_filter(email):
    """
    Rule-based spam detection
    """
    spam_words = ['free', 'winner', 'cash', 'prize', 'click here']
    email_lower = email.lower()
    
    for word in spam_words:
        if word in email_lower:
            return "SPAM"
    return "NOT SPAM"

# Machine Learning Approach
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
emails = [
    "Win free cash now!",
    "Meeting at 3pm tomorrow",
    "Click here for prize",
    "Project deadline reminder",
    "Congratulations! You're a winner!",
    "Can we reschedule our call?"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

# ML approach
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
classifier = MultinomialNB()
classifier.fit(X, labels)

# Compare approaches
test_email = "free money winner"
print(f"Traditional: {traditional_spam_filter(test_email)}")
test_vector = vectorizer.transform([test_email])
ml_prediction = classifier.predict(test_vector)[0]
print(f"ML Approach: {'SPAM' if ml_prediction == 1 else 'NOT SPAM'}")
```

### When to Use Machine Learning?

Machine Learning is ideal when:
1. **Pattern exists**: There's a pattern in the data
2. **Cannot pin down mathematically**: Pattern is complex
3. **Have data**: Sufficient data is available
4. **Predictions are valuable**: The task is worth automating

---

## Types of Machine Learning

### 1. Supervised Learning

Learning from labeled examples (input-output pairs).

![Supervised Learning](https://miro.medium.com/max/1400/1*-fniNC4C8WDWG8H4NzVBJA.png)

```python
# Supervised Learning Example
from sklearn.datasets import make_classification, make_regression

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Classification
X_class, y_class = make_classification(n_samples=100, n_features=2, 
                                       n_redundant=0, n_clusters_per_class=1,
                                       random_state=42)
ax = axes[0]
scatter = ax.scatter(X_class[:, 0], X_class[:, 1], c=y_class, 
                     cmap='viridis', edgecolor='k', s=50)
ax.set_title('Classification Task')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
plt.colorbar(scatter, ax=ax)

# Regression
X_reg = np.linspace(0, 10, 100).reshape(-1, 1)
y_reg = 2 * X_reg.squeeze() + 1 + np.random.normal(0, 2, 100)
ax = axes[1]
ax.scatter(X_reg, y_reg, alpha=0.5)
ax.plot(X_reg, 2 * X_reg + 1, 'r-', linewidth=2, label='True function')
ax.set_title('Regression Task')
ax.set_xlabel('Feature')
ax.set_ylabel('Target')
ax.legend()

plt.tight_layout()
plt.show()
```

**Common Algorithms:**
- **Classification**: Logistic Regression, SVM, Decision Trees, Random Forest
- **Regression**: Linear Regression, Polynomial Regression, Ridge, Lasso

### 2. Unsupervised Learning

Finding patterns in unlabeled data.

```python
# Unsupervised Learning Example
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before clustering (no labels)
ax1.scatter(X[:, 0], X[:, 1], alpha=0.5, s=50)
ax1.set_title('Unlabeled Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# After clustering
ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
           edgecolor='k', s=50)
ax2.scatter(kmeans.cluster_centers_[:, 0], 
           kmeans.cluster_centers_[:, 1],
           c='red', marker='x', s=200, linewidths=3)
ax2.set_title('K-Means Clustering Result')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

**Common Algorithms:**
- **Clustering**: K-Means, DBSCAN, Hierarchical Clustering
- **Dimensionality Reduction**: PCA, t-SNE, Autoencoders
- **Anomaly Detection**: Isolation Forest, One-Class SVM

### 3. Reinforcement Learning

Learning through interaction with environment.

![Reinforcement Learning](https://miro.medium.com/max/1400/1*7cuAqjQ97x1H_sBIeAVVZg.png)

```python
# Simple Reinforcement Learning Concept
class SimpleGridWorld:
    def __init__(self):
        self.state = 0
        self.goal = 9
        
    def step(self, action):
        # action: 0=left, 1=right
        if action == 1 and self.state < self.goal:
            self.state += 1
            reward = 1 if self.state == self.goal else -0.1
        elif action == 0 and self.state > 0:
            self.state -= 1
            reward = -0.1
        else:
            reward = -1  # Invalid action
            
        done = self.state == self.goal
        return self.state, reward, done
    
    def reset(self):
        self.state = 0
        return self.state

# Q-learning agent (simplified)
class QLearningAgent:
    def __init__(self, n_states=10, n_actions=2):
        self.q_table = np.zeros((n_states, n_actions))
        self.learning_rate = 0.1
        self.discount = 0.95
        self.epsilon = 0.1
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(2)  # Explore
        return np.argmax(self.q_table[state])  # Exploit
    
    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q)
        self.q_table[state, action] = new_q

# Training loop visualization
env = SimpleGridWorld()
agent = QLearningAgent()
rewards_history = []

for episode in range(100):
    state = env.reset()
    total_reward = 0
    
    for _ in range(20):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    rewards_history.append(total_reward)

# Plot learning curve
plt.figure(figsize=(10, 5))
plt.plot(rewards_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reinforcement Learning: Agent Performance Over Time')
plt.grid(True, alpha=0.3)
plt.show()

# Show learned Q-table
print("Learned Q-values:")
print("State | Left | Right")
for i, row in enumerate(agent.q_table):
    print(f"  {i}   | {row[0]:.2f} | {row[1]:.2f}")
```

### 4. Semi-Supervised Learning

Combination of labeled and unlabeled data.

```python
from sklearn.semi_supervised import LabelPropagation

# Generate data with few labels
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# Hide most labels (simulate semi-supervised scenario)
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(y)) < 0.9
labels = np.copy(y)
labels[random_unlabeled_points] = -1  # -1 indicates unlabeled

# Apply semi-supervised learning
label_prop = LabelPropagation()
label_prop.fit(X, labels)
predicted_labels = label_prop.predict(X)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original labels (mostly hidden)
ax1.scatter(X[labels == 0, 0], X[labels == 0, 1], c='blue', 
           label='Class 0', s=50, edgecolor='k')
ax1.scatter(X[labels == 1, 0], X[labels == 1, 1], c='red', 
           label='Class 1', s=50, edgecolor='k')
ax1.scatter(X[labels == -1, 0], X[labels == -1, 1], c='gray', 
           label='Unlabeled', alpha=0.3, s=20)
ax1.set_title('Semi-Supervised: Input Data')
ax1.legend()

# After label propagation
ax2.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='coolwarm',
           edgecolor='k', s=50)
ax2.set_title('After Label Propagation')

plt.tight_layout()
plt.show()

print(f"Labeled samples: {np.sum(labels != -1)}")
print(f"Unlabeled samples: {np.sum(labels == -1)}")
print(f"Accuracy on originally labeled: {accuracy_score(y[labels != -1], predicted_labels[labels != -1]):.3f}")
```

---

## The Machine Learning Pipeline

![ML Pipeline](https://miro.medium.com/max/1400/0*4BjxmMFIJiYZ9KmF.png)

### Complete Pipeline Example

```python
# Complete ML Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# 1. Data Collection & Preparation
from sklearn.datasets import load_wine
data = load_wine()
X, y = data.data, data.target

print("=" * 50)
print("MACHINE LEARNING PIPELINE DEMONSTRATION")
print("=" * 50)
print(f"\n1. DATA COLLECTION")
print(f"   Dataset: Wine Classification")
print(f"   Samples: {X.shape[0]}")
print(f"   Features: {X.shape[1]}")
print(f"   Classes: {len(np.unique(y))}")

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n2. DATA SPLITTING")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# 3. Create Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 4. Hyperparameter Tuning
param_grid = {
    'pca__n_components': [5, 8, 10],
    'classifier__C': [0.1, 1, 10]
}

print(f"\n3. MODEL TRAINING & HYPERPARAMETER TUNING")
print(f"   Pipeline steps: {[name for name, _ in pipeline.steps]}")
print(f"   Parameter grid: {list(param_grid.keys())}")

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"\n4. BEST PARAMETERS FOUND")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

# 5. Evaluation
train_score = grid_search.score(X_train, y_train)
test_score = grid_search.score(X_test, y_test)

print(f"\n5. MODEL EVALUATION")
print(f"   Training Accuracy: {train_score:.3f}")
print(f"   Test Accuracy: {test_score:.3f}")

# 6. Visualize Pipeline Performance
results_df = pd.DataFrame(grid_search.cv_results_)
pivot_table = results_df.pivot_table(
    values='mean_test_score',
    index='param_classifier__C',
    columns='param_pca__n_components'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('Grid Search Results: Accuracy Heatmap')
plt.xlabel('PCA Components')
plt.ylabel('Regularization Strength (C)')
plt.show()
```

---

## Training, Validation, and Test Sets

### Why Split Data?

```python
# Demonstrate importance of train-test split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X.squeeze() + 1 + np.random.normal(0, 3, 100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Try different polynomial degrees
degrees = [1, 3, 9, 15]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, degree in zip(axes.flat, degrees):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predictions
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    
    # Scores
    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)
    
    # Plot
    ax.scatter(X_train, y_train, alpha=0.5, label='Train data')
    ax.scatter(X_test, y_test, alpha=0.5, label='Test data')
    ax.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'Degree {degree}')
    ax.set_title(f'Polynomial Degree: {degree}\nTrain R²: {train_score:.3f}, Test R²: {test_score:.3f}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_ylim([-10, 30])

plt.tight_layout()
plt.show()
```

### Data Splitting Strategies

```python
from sklearn.model_selection import StratifiedShuffleSplit, TimeSeriesSplit

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 1. Random Split
ax = axes[0]
X = np.arange(100)
y = np.random.randint(0, 2, 100)
train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)

ax.scatter(train_idx, np.ones(len(train_idx)), c='blue', label='Train', s=20)
ax.scatter(test_idx, np.ones(len(test_idx)) * 1.1, c='red', label='Test', s=20)
ax.set_title('Random Train-Test Split')
ax.set_ylim([0.9, 1.2])
ax.legend()

# 2. Stratified Split (preserves class distribution)
ax = axes[1]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    ax.scatter(train_idx[y[train_idx] == 0], np.ones(sum(y[train_idx] == 0)), 
              c='blue', marker='o', label='Train Class 0', s=20)
    ax.scatter(train_idx[y[train_idx] == 1], np.ones(sum(y[train_idx] == 1)) * 1.05, 
              c='darkblue', marker='^', label='Train Class 1', s=20)
    ax.scatter(test_idx[y[test_idx] == 0], np.ones(sum(y[test_idx] == 0)) * 1.1, 
              c='red', marker='o', label='Test Class 0', s=20)
    ax.scatter(test_idx[y[test_idx] == 1], np.ones(sum(y[test_idx] == 1)) * 1.15, 
              c='darkred', marker='^', label='Test Class 1', s=20)
ax.set_title('Stratified Split (Preserves Class Distribution)')
ax.set_ylim([0.95, 1.2])
ax.legend(ncol=2)

# 3. Time Series Split
ax = axes[2]
tscv = TimeSeriesSplit(n_splits=5)
for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
    ax.scatter(train_idx, np.ones(len(train_idx)) * i, c='blue', s=10)
    ax.scatter(test_idx, np.ones(len(test_idx)) * i, c='red', s=10)

ax.set_title('Time Series Cross-Validation')
ax.set_ylabel('Fold')
ax.set_xlabel('Sample Index')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', label='Train'),
                  Patch(facecolor='red', label='Test')]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.show()
```

---

## Overfitting and Underfitting

### Understanding the Problem

```python
# Generate dataset
np.random.seed(42)
n_samples = 100
X = np.sort(np.random.rand(n_samples))
y = np.sin(2 * np.pi * X) + np.random.randn(n_samples) * 0.1

X = X.reshape(-1, 1)

# Create models with different complexity
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Different model complexities
complexities = [
    ('Underfitting', 1, 'blue'),
    ('Good Fit', 4, 'green'),
    ('Overfitting', 15, 'red')
]

X_test = np.linspace(0, 1, 300).reshape(-1, 1)

for ax, (title, degree, color) in zip(axes, complexities):
    # Create and train model
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        Ridge(alpha=0.001)
    )
    model.fit(X, y)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate error
    train_score = model.score(X, y)
    
    # Plot
    ax.scatter(X, y, alpha=0.5, s=20, label='Training data')
    ax.plot(X_test, y_pred, color=color, linewidth=2, 
           label=f'Polynomial degree {degree}')
    ax.plot(X_test, np.sin(2 * np.pi * X_test), 'k--', 
           alpha=0.5, linewidth=1, label='True function')
    ax.set_title(f'{title}\nTrain Score: {train_score:.3f}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_ylim([-1.5, 1.5])

plt.tight_layout()
plt.show()
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, title):
    """
    Plot learning curves to diagnose bias/variance
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    train_scores = -train_scores
    val_scores = -val_scores
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'b-', 
            label='Training error', linewidth=2)
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'r-', 
            label='Validation error', linewidth=2)
    
    plt.fill_between(train_sizes, 
                    np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                    np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                    alpha=0.2, color='blue')
    plt.fill_between(train_sizes,
                    np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                    np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                    alpha=0.2, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Compare different models
models = [
    ('High Bias (Underfitting)', make_pipeline(PolynomialFeatures(1), Ridge())),
    ('Good Balance', make_pipeline(PolynomialFeatures(4), Ridge())),
    ('High Variance (Overfitting)', make_pipeline(PolynomialFeatures(15), Ridge(alpha=0.001)))
]

for title, model in models:
    plot_learning_curves(model, X, y, title)
```

---

## Bias-Variance Tradeoff

### Mathematical Decomposition

Expected Error = Bias² + Variance + Irreducible Error

```python
# Simulate bias-variance tradeoff
def bias_variance_simulation(n_simulations=100):
    """
    Simulate bias-variance tradeoff with different model complexities
    """
    np.random.seed(42)
    
    # True function
    def true_function(x):
        return np.sin(2 * np.pi * x)
    
    # Generate test point
    x_test = 0.5
    y_true = true_function(x_test)
    
    complexities = range(1, 16)
    bias_squared = []
    variance = []
    
    for degree in complexities:
        predictions = []
        
        for _ in range(n_simulations):
            # Generate new training data
            X_train = np.random.rand(20).reshape(-1, 1)
            noise = np.random.randn(20) * 0.1
            y_train = true_function(X_train.ravel()) + noise
            
            # Train model
            model = make_pipeline(
                PolynomialFeatures(degree),
                Ridge(alpha=0.01)
            )
            model.fit(X_train, y_train)
            
            # Predict at test point
            pred = model.predict([[x_test]])[0]
            predictions.append(pred)
        
        # Calculate bias and variance
        predictions = np.array(predictions)
        bias = np.mean(predictions) - y_true
        bias_squared.append(bias ** 2)
        variance.append(np.var(predictions))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Components
    ax1.plot(complexities, bias_squared, 'b-', label='Bias²', linewidth=2)
    ax1.plot(complexities, variance, 'r-', label='Variance', linewidth=2)
    ax1.plot(complexities, np.array(bias_squared) + np.array(variance), 
            'g-', label='Total Error', linewidth=2)
    ax1.set_xlabel('Model Complexity (Polynomial Degree)')
    ax1.set_ylabel('Error')
    ax1.set_title('Bias-Variance Decomposition')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Tradeoff visualization
    ax2.fill_between(complexities, 0, bias_squared, alpha=0.3, color='blue', label='Bias²')
    ax2.fill_between(complexities, bias_squared, 
                     np.array(bias_squared) + np.array(variance), 
                     alpha=0.3, color='red', label='Variance')
    ax2.set_xlabel('Model Complexity')
    ax2.set_ylabel('Error Components')
    ax2.set_title('Bias-Variance Tradeoff')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal complexity
    total_error = np.array(bias_squared) + np.array(variance)
    optimal_degree = complexities[np.argmin(total_error)]
    print(f"Optimal polynomial degree: {optimal_degree}")
    
    return bias_squared, variance

# Run simulation
bias_squared, variance = bias_variance_simulation()
```

---

## Model Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_curve, auc, classification_report)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, n_classes=2, weights=[0.9, 0.1],
                          random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Classification Metrics:")
print("=" * 40)
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Confusion Matrix
ax = axes[0, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

# ROC Curve
ax = axes[0, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()

# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
ax = axes[1, 0]
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
ax.plot(recall_vals, precision_vals, 'g-', linewidth=2)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.fill_between(recall_vals, precision_vals, alpha=0.3)

# Threshold Analysis
ax = axes[1, 1]
thresholds = np.linspace(0, 1, 100)
metrics = {'precision': [], 'recall': [], 'f1': []}

for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    if np.sum(y_pred_thresh) > 0:  # Avoid division by zero
        metrics['precision'].append(precision_score(y_test, y_pred_thresh, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred_thresh))
        metrics['f1'].append(f1_score(y_test, y_pred_thresh))
    else:
        metrics['precision'].append(0)
        metrics['recall'].append(0)
        metrics['f1'].append(0)

ax.plot(thresholds, metrics['precision'], label='Precision')
ax.plot(thresholds, metrics['recall'], label='Recall')
ax.plot(thresholds, metrics['f1'], label='F1-Score')
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Metric Value')
ax.set_title('Metrics vs Decision Threshold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Regression Metrics

```python
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                           r2_score, mean_absolute_percentage_error)
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# Generate regression data
X, y = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Regression Metrics:")
print("=" * 40)
print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R²:   {r2:.3f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Predicted vs Actual
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
       'r--', linewidth=2)
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title(f'Predicted vs Actual (R² = {r2:.3f})')

# Residuals
ax = axes[0, 1]
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, alpha=0.5)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')

# Residual Distribution
ax = axes[1, 0]
ax.hist(residuals, bins=30, edgecolor='black')
ax.set_xlabel('Residuals')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution')

# Q-Q Plot
from scipy import stats
ax = axes[1, 1]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Check Normality of Residuals)')

plt.tight_layout()
plt.show()
```

---

## Cross-Validation

### Different CV Strategies

```python
from sklearn.model_selection import (KFold, StratifiedKFold, LeaveOneOut, 
                                     cross_val_score, cross_validate)

# Generate data
X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

# Different CV strategies
cv_strategies = {
    'K-Fold (k=5)': KFold(n_splits=5, shuffle=True, random_state=42),
    'Stratified K-Fold (k=5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'Leave-One-Out': LeaveOneOut()
}

# Compare strategies
model = LogisticRegression(max_iter=1000)
results = {}

for name, cv in cv_strategies.items():
    if name != 'Leave-One-Out':  # LOO is too slow for visualization
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        results[name] = scores
        print(f"{name:25} Mean: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

# Visualize fold distributions
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# K-Fold visualization
ax = axes[0]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_idx, val_idx) in enumerate(kf.split(X)):
    ax.scatter(train_idx, np.ones(len(train_idx)) * i, c='blue', s=10, alpha=0.5)
    ax.scatter(val_idx, np.ones(len(val_idx)) * i, c='red', s=10, alpha=0.5)
ax.set_title('K-Fold Cross-Validation')
ax.set_ylabel('Fold')
ax.set_xlabel('Sample Index')

# Stratified K-Fold visualization
ax = axes[1]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Color by class
    train_class_0 = train_idx[y[train_idx] == 0]
    train_class_1 = train_idx[y[train_idx] == 1]
    val_class_0 = val_idx[y[val_idx] == 0]
    val_class_1 = val_idx[y[val_idx] == 1]
    
    ax.scatter(train_class_0, np.ones(len(train_class_0)) * i, 
              c='lightblue', s=10, alpha=0.5)
    ax.scatter(train_class_1, np.ones(len(train_class_1)) * i, 
              c='darkblue', s=10, alpha=0.5)
    ax.scatter(val_class_0, np.ones(len(val_class_0)) * i, 
              c='lightcoral', s=10, alpha=0.5)
    ax.scatter(val_class_1, np.ones(len(val_class_1)) * i, 
              c='darkred', s=10, alpha=0.5)

ax.set_title('Stratified K-Fold Cross-Validation (Preserves Class Distribution)')
ax.set_ylabel('Fold')
ax.set_xlabel('Sample Index')

plt.tight_layout()
plt.show()

# Cross-validation with multiple metrics
scoring = ['accuracy', 'precision', 'recall', 'f1']
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

print("\nDetailed Cross-Validation Results:")
print("=" * 50)
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric:10} {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
```

---

## Feature Engineering

### Feature Creation and Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import PolynomialFeatures

# Generate data with informative and noise features
X, y = make_classification(n_samples=200, n_features=20, n_informative=5,
                          n_redundant=5, n_repeated=2, n_classes=2,
                          random_state=42)

feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

# 1. Univariate Feature Selection
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
scores = selector.scores_

# 2. Recursive Feature Elimination
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)

# 3. Feature Importance from Tree-based Model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_

# Visualize feature selection results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Univariate scores
ax = axes[0, 0]
indices = np.argsort(scores)[::-1][:10]
ax.bar(range(10), scores[indices])
ax.set_xticks(range(10))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
ax.set_title('Top 10 Features (Univariate Selection)')
ax.set_ylabel('F-Score')

# RFE ranking
ax = axes[0, 1]
rfe_scores = 1 / rfe.ranking_
indices = np.argsort(rfe_scores)[::-1][:10]
ax.bar(range(10), rfe_scores[indices])
ax.set_xticks(range(10))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
ax.set_title('Top 10 Features (RFE)')
ax.set_ylabel('RFE Score')

# Feature importance
ax = axes[1, 0]
indices = np.argsort(importances)[::-1][:10]
ax.bar(range(10), importances[indices])
ax.set_xticks(range(10))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
ax.set_title('Top 10 Features (Random Forest)')
ax.set_ylabel('Importance')

# Correlation matrix
ax = axes[1, 1]
correlation_matrix = np.corrcoef(X.T)
im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

### Feature Transformation

```python
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, 
                                   RobustScaler, PowerTransformer)

# Generate data with different distributions
np.random.seed(42)
n_samples = 1000

# Different feature distributions
normal_feature = np.random.normal(100, 20, n_samples)
skewed_feature = np.random.exponential(2, n_samples)
uniform_feature = np.random.uniform(0, 100, n_samples)
outlier_feature = np.concatenate([np.random.normal(50, 5, 950),
                                  np.random.normal(150, 5, 50)])

X_original = np.column_stack([normal_feature, skewed_feature, 
                              uniform_feature, outlier_feature])

# Apply different scalers
scalers = {
    'Original': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'PowerTransformer': PowerTransformer()
}

fig, axes = plt.subplots(len(scalers), 4, figsize=(16, 15))

for i, (name, scaler) in enumerate(scalers.items()):
    if scaler is None:
        X_transformed = X_original
    else:
        X_transformed = scaler.fit_transform(X_original)
    
    for j in range(4):
        ax = axes[i, j]
        ax.hist(X_transformed[:, j], bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'{name}\nFeature {j+1}')
        ax.set_ylabel('Frequency')
        
        # Add statistics
        mean = np.mean(X_transformed[:, j])
        std = np.std(X_transformed[:, j])
        ax.text(0.02, 0.98, f'μ={mean:.2f}\nσ={std:.2f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
```

---

## Practical Examples

### Example 1: Complete Classification Pipeline

```python
# Real-world example: Iris classification
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

print("Complete Classification Pipeline Example")
print("=" * 50)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', SVC())
])

# Define parameter grid
param_grid = {
    'pca__n_components': [2, 3, 4],
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1
)

# Fit the model
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Evaluate final model
best_model = grid_search.best_estimator_
scores = cross_val_score(best_model, X, y, cv=10)
print(f"10-fold CV scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

# Visualize decision boundaries (using only 2 principal components)
best_model.set_params(pca__n_components=2)
best_model.fit(X, y)
X_transformed = best_model.named_steps['scaler'].transform(X)
X_transformed = best_model.named_steps['pca'].transform(X_transformed)

# Create mesh
h = 0.02
x_min, x_max = X_transformed[:, 0].min() - 1, X_transformed[:, 0].max() + 1
y_min, y_max = X_transformed[:, 1].min() - 1, X_transformed[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict on mesh
Z = best_model.named_steps['classifier'].predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                     c=y, cmap='viridis', edgecolor='black', s=50)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('SVM Decision Boundaries in PCA Space')
plt.colorbar(scatter)
plt.show()
```

### Example 2: Regression with Feature Engineering

```python
# Boston housing price prediction
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("Gradient Boosting Regression Results")
print("=" * 50)
print(f"Training R²: {train_score:.3f}")
print(f"Test R²: {test_score:.3f}")

# Feature importance
importance = model.feature_importances_
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Built-in feature importance
ax = axes[0]
indices = np.argsort(importance)[::-1]
ax.barh(range(len(importance)), importance[indices])
ax.set_yticks(range(len(importance)))
ax.set_yticklabels([feature_names[i] for i in indices])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance (MDI)')

# Permutation importance
ax = axes[1]
indices = np.argsort(perm_importance.importances_mean)[::-1]
ax.barh(range(len(indices)), perm_importance.importances_mean[indices])
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices])
ax.set_xlabel('Importance')
ax.set_title('Permutation Importance')

plt.tight_layout()
plt.show()

# Partial dependence plots
from sklearn.inspection import PartialDependenceDisplay

features = [0, 5, (0, 5)]  # Individual and interaction effects
display = PartialDependenceDisplay.from_estimator(
    model, X_train, features, feature_names=feature_names
)
display.figure_.suptitle('Partial Dependence Plots')
plt.show()
```

---

## Summary and Key Takeaways

### 🎯 Core Concepts Mastered

1. **ML Fundamentals**
   - Difference between traditional programming and ML
   - When to use machine learning
   - Types of learning problems

2. **ML Pipeline**
   - Data collection and preparation
   - Train-validation-test splitting
   - Model training and evaluation
   - Deployment considerations

3. **Model Complexity**
   - Overfitting vs underfitting
   - Bias-variance tradeoff
   - Regularization techniques

4. **Evaluation**
   - Classification metrics (accuracy, precision, recall, F1)
   - Regression metrics (MSE, MAE, R²)
   - Cross-validation strategies

5. **Feature Engineering**
   - Feature selection methods
   - Feature transformation and scaling
   - Creating new features

### 💡 Key Insights

1. **No Free Lunch**: No single algorithm works best for all problems
2. **Data Quality > Algorithm Complexity**: Better data beats fancier algorithms
3. **Validation is Crucial**: Always evaluate on unseen data
4. **Iterative Process**: ML is experimental and iterative
5. **Domain Knowledge Matters**: Understanding the problem domain improves results

[**Continue to Module 4: Supervised Learning →**](04_supervised_learning.md)