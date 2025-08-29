# Module 8: Decision Trees and Random Forests 🌲

[← Previous Module](07_scikit_learn.md) | [Back to Main](../README.md)

## 📋 Table of Contents
1. [Introduction to Decision Trees](#introduction-to-decision-trees)
2. [How Decision Trees Work](#how-decision-trees-work)
3. [Information Gain and Gini Index](#information-gain-and-gini-index)
4. [Building Decision Trees](#building-decision-trees)
5. [Tree Pruning and Regularization](#tree-pruning-and-regularization)
6. [Ensemble Methods Overview](#ensemble-methods-overview)
7. [Random Forests in Detail](#random-forests-in-detail)
8. [Feature Importance](#feature-importance)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Advanced Techniques](#advanced-techniques)
11. [Real-World Applications](#real-world-applications)
12. [Summary and Best Practices](#summary-and-best-practices)

---

## Introduction to Decision Trees

Decision Trees are powerful, interpretable machine learning algorithms that make decisions by asking a series of questions about the features.

![Decision Tree Concept](https://miro.medium.com/max/1400/1*XMId5sJqPtm8-RIwVVz2tg.png)

### Why Decision Trees?

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Advantages of Decision Trees
advantages = {
    "Interpretability": "Easy to understand and visualize",
    "No preprocessing": "Handles numerical and categorical data",
    "Non-linear": "Captures non-linear relationships",
    "Feature selection": "Automatic feature selection",
    "Robust": "Handles outliers well"
}

print("DECISION TREES: KEY ADVANTAGES")
print("="*50)
for key, value in advantages.items():
    print(f"✓ {key}: {value}")
```

### Simple Example

```python
# Create a simple dataset
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                          n_clusters_per_class=1, random_state=42)

# Train decision tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# Visualize decision boundary
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot decision boundary
ax = axes[0]
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
          edgecolor='black', s=50)
ax.set_title('Decision Tree: Decision Boundary')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# Plot tree structure
ax = axes[1]
plot_tree(dt, filled=True, ax=ax, feature_names=['F1', 'F2'])
ax.set_title('Decision Tree Structure')

plt.tight_layout()
plt.show()
```

---

## How Decision Trees Work

### The Decision Process

```python
# Demonstrate decision tree logic
class SimpleDecisionTree:
    """Simplified decision tree for illustration"""
    
    def __init__(self):
        self.tree = None
    
    def make_decision(self, sample):
        """Walk through decision tree"""
        decisions = []
        
        # Example decision path
        if sample[0] <= 0.5:
            decisions.append("Feature 1 <= 0.5")
            if sample[1] <= 0.3:
                decisions.append("Feature 2 <= 0.3")
                prediction = "Class A"
            else:
                decisions.append("Feature 2 > 0.3")
                prediction = "Class B"
        else:
            decisions.append("Feature 1 > 0.5")
            if sample[1] <= 0.7:
                decisions.append("Feature 2 <= 0.7")
                prediction = "Class B"
            else:
                decisions.append("Feature 2 > 0.7")
                prediction = "Class A"
        
        return prediction, decisions

# Example usage
tree = SimpleDecisionTree()
sample = [0.3, 0.6]
prediction, path = tree.make_decision(sample)

print("DECISION TREE LOGIC")
print("="*50)
print(f"Sample: {sample}")
print(f"Decision path:")
for i, decision in enumerate(path, 1):
    print(f"  {i}. {decision}")
print(f"Final prediction: {prediction}")
```

### Tree Growing Algorithm

```python
def visualize_tree_growth():
    """Visualize how a tree grows with increasing depth"""
    
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                              n_clusters_per_class=1, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    depths = [1, 2, 3, 5, 10, None]
    
    for ax, depth in zip(axes, depths):
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X, y)
        
        # Decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', 
                  edgecolor='black', s=20)
        
        score = dt.score(X, y)
        depth_str = 'No limit' if depth is None else str(depth)
        ax.set_title(f'Max Depth: {depth_str}\nAccuracy: {score:.3f}')
        
    plt.suptitle('Decision Tree Growth with Increasing Depth', fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_tree_growth()
```

---

## Information Gain and Gini Index

### Splitting Criteria

```python
def entropy(y):
    """Calculate entropy of a label array"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def gini_impurity(y):
    """Calculate Gini impurity of a label array"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

def information_gain(parent, left_child, right_child):
    """Calculate information gain of a split"""
    num_left = len(left_child)
    num_right = len(right_child)
    num_parent = num_left + num_right
    
    # Weighted average of child entropies
    weighted_entropy = (num_left/num_parent * entropy(left_child) + 
                       num_right/num_parent * entropy(right_child))
    
    # Information gain
    return entropy(parent) - weighted_entropy

# Example calculation
parent = np.array([0, 0, 0, 1, 1, 1, 1, 1])
left_child = np.array([0, 0, 0, 1])
right_child = np.array([1, 1, 1, 1])

print("SPLITTING CRITERIA")
print("="*50)
print(f"Parent entropy: {entropy(parent):.3f}")
print(f"Parent Gini: {gini_impurity(parent):.3f}")
print(f"Left child entropy: {entropy(left_child):.3f}")
print(f"Right child entropy: {entropy(right_child):.3f}")
print(f"Information gain: {information_gain(parent, left_child, right_child):.3f}")
```

### Visualizing Split Quality

```python
def visualize_splits():
    """Visualize different split qualities"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Different split scenarios
    splits = [
        ("Perfect Split", [0,0,0,0,0], [1,1,1,1,1]),
        ("Good Split", [0,0,0,0,1], [0,1,1,1,1]),
        ("Moderate Split", [0,0,0,1,1], [0,0,1,1,1]),
        ("Poor Split", [0,0,1,1,1], [0,0,0,1,1]),
        ("Random Split", [0,1,0,1,0], [1,0,1,0,1]),
        ("No Split", [0,0,1,1,1], [0,0,1,1,1])
    ]
    
    for ax, (title, left, right) in zip(axes.flat, splits):
        left = np.array(left)
        right = np.array(right)
        parent = np.concatenate([left, right])
        
        # Calculate metrics
        gini_parent = gini_impurity(parent)
        gini_left = gini_impurity(left)
        gini_right = gini_impurity(right)
        
        weighted_gini = (len(left)/len(parent) * gini_left + 
                        len(right)/len(parent) * gini_right)
        gini_gain = gini_parent - weighted_gini
        
        # Visualize
        x = np.arange(len(parent))
        colors = ['blue' if i < len(left) else 'red' for i in range(len(parent))]
        ax.bar(x, parent, color=colors, alpha=0.6)
        ax.axvline(len(left)-0.5, color='black', linestyle='--', linewidth=2)
        
        ax.set_title(f'{title}\nGini Gain: {gini_gain:.3f}')
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel('Samples')
        ax.set_ylabel('Class')
        
        # Add text
        ax.text(len(left)/2, 0.5, f'Left\nGini: {gini_left:.2f}', 
               ha='center', va='center', fontsize=10)
        ax.text(len(left) + len(right)/2, 0.5, f'Right\nGini: {gini_right:.2f}', 
               ha='center', va='center', fontsize=10)
    
    plt.suptitle('Split Quality Visualization', fontsize=16)
    plt.tight_layout()
    plt.show()

visualize_splits()
```

---

## Building Decision Trees

### Classification Trees

```python
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Build classification tree
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)

# Evaluate
train_score = clf_tree.score(X_train, y_train)
test_score = clf_tree.score(X_test, y_test)

print("CLASSIFICATION TREE RESULTS")
print("="*50)
print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")
print(f"Tree depth: {clf_tree.get_depth()}")
print(f"Number of leaves: {clf_tree.get_n_leaves()}")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(clf_tree, feature_names=iris.feature_names,
         class_names=iris.target_names, filled=True,
         rounded=True, fontsize=10)
plt.title("Iris Classification Tree")
plt.show()

# Feature importance
importances = clf_tree.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [iris.feature_names[i] for i in indices])
plt.title("Feature Importance - Classification Tree")
plt.tight_layout()
plt.show()
```

### Regression Trees

```python
from sklearn.datasets import fetch_california_housing

# Load housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Sample subset for visualization
sample_idx = np.random.choice(len(X), 1000, replace=False)
X_sample = X[sample_idx]
y_sample = y[sample_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.3, random_state=42
)

# Build regression tree
reg_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
reg_tree.fit(X_train, y_train)

# Predictions
y_pred = reg_tree.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = reg_tree.score(X_test, y_test)

print("\nREGRESSION TREE RESULTS")
print("="*50)
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"Tree depth: {reg_tree.get_depth()}")
print(f"Number of leaves: {reg_tree.get_n_leaves()}")

# Visualize predictions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Actual vs Predicted
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
       'r--', linewidth=2)
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
ax.set_title(f'Regression Tree: Actual vs Predicted\nR² = {r2:.3f}')

# Residuals
ax = axes[1]
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, alpha=0.5)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Predicted Price')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')

plt.tight_layout()
plt.show()
```

---

## Tree Pruning and Regularization

### Preventing Overfitting

```python
def compare_pruning_methods():
    """Compare different pruning strategies"""
    
    X, y = make_classification(n_samples=500, n_features=20, 
                              n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Different regularization parameters
    params = [
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': None, 'min_samples_split': 20, 'min_samples_leaf': 1},
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 10},
        {'max_depth': 3, 'min_samples_split': 10, 'min_samples_leaf': 5}
    ]
    
    results = []
    
    for param in params:
        dt = DecisionTreeClassifier(**param, random_state=42)
        dt.fit(X_train, y_train)
        
        train_score = dt.score(X_train, y_train)
        test_score = dt.score(X_test, y_test)
        
        results.append({
            'params': str(param),
            'train_score': train_score,
            'test_score': test_score,
            'depth': dt.get_depth(),
            'n_leaves': dt.get_n_leaves(),
            'overfit': train_score - test_score
        })
    
    results_df = pd.DataFrame(results)
    
    print("PRUNING COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results))
    width = 0.35
    
    ax.bar(x - width/2, results_df['train_score'], width, 
          label='Train', alpha=0.8)
    ax.bar(x + width/2, results_df['test_score'], width, 
          label='Test', alpha=0.8)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Effect of Pruning on Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Config {i+1}' for i in range(len(results))], 
                      rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

pruning_results = compare_pruning_methods()
```

### Cost Complexity Pruning

```python
# Cost complexity pruning (minimal cost-complexity pruning)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Get the pruning path
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train trees with different alpha values
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Evaluate
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
tree_depths = [clf.get_depth() for clf in clfs]

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Impurity vs alpha
ax = axes[0]
ax.plot(ccp_alphas, impurities, marker='o')
ax.set_xlabel('Alpha')
ax.set_ylabel('Total Impurity of Leaves')
ax.set_title('Total Impurity vs Alpha')

# Accuracy vs alpha
ax = axes[1]
ax.plot(ccp_alphas, train_scores, marker='o', label='Train')
ax.plot(ccp_alphas, test_scores, marker='o', label='Test')
ax.set_xlabel('Alpha')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Alpha')
ax.legend()

# Tree depth vs alpha
ax = axes[2]
ax.plot(ccp_alphas, tree_depths, marker='o')
ax.set_xlabel('Alpha')
ax.set_ylabel('Tree Depth')
ax.set_title('Tree Depth vs Alpha')

plt.tight_layout()
plt.show()

# Find optimal alpha
optimal_idx = np.argmax(test_scores)
optimal_alpha = ccp_alphas[optimal_idx]
print(f"\nOptimal alpha: {optimal_alpha:.4f}")
print(f"Test accuracy with optimal alpha: {test_scores[optimal_idx]:.3f}")
```

---

## Ensemble Methods Overview

### Why Ensemble Methods?

```python
# Demonstrate ensemble concept
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Individual models
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Train and evaluate individual models
print("ENSEMBLE METHODS: INDIVIDUAL MODEL PERFORMANCE")
print("="*50)

individual_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    individual_scores[name] = score
    print(f"{name:20} Accuracy: {score:.3f}")

# Create ensemble
ensemble = VotingClassifier(
    estimators=list(models.items()),
    voting='soft'
)
ensemble.fit(X_train, y_train)
ensemble_score = ensemble.score(X_test, y_test)

print(f"\n{'Ensemble (Voting)':20} Accuracy: {ensemble_score:.3f}")
print(f"Improvement: {ensemble_score - np.mean(list(individual_scores.values())):.3f}")
```

### Types of Ensemble Methods

```python
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier, 
                             GradientBoostingClassifier)

# Different ensemble methods
ensembles = {
    'Bagging': BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=100, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100, random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42
    )
}

# Compare ensemble methods
results = []

for name, ensemble in ensembles.items():
    # Train
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    train_score = ensemble.score(X_train, y_train)
    test_score = ensemble.score(X_test, y_test)
    
    results.append({
        'Method': name,
        'Train Accuracy': train_score,
        'Test Accuracy': test_score,
        'Overfit': train_score - test_score
    })

results_df = pd.DataFrame(results)

print("\nENSEMBLE METHODS COMPARISON")
print("="*60)
print(results_df.to_string(index=False))

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.35

ax.bar(x - width/2, results_df['Train Accuracy'], width, 
      label='Train', alpha=0.8)
ax.bar(x + width/2, results_df['Test Accuracy'], width, 
      label='Test', alpha=0.8)

ax.set_xlabel('Ensemble Method')
ax.set_ylabel('Accuracy')
ax.set_title('Ensemble Methods Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Method'])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Random Forests in Detail

### How Random Forests Work

```python
class RandomForestDemonstration:
    """Demonstrate Random Forest concepts"""
    
    def __init__(self, n_trees=5):
        self.n_trees = n_trees
        
    def demonstrate_bootstrap(self, X, y):
        """Show bootstrap sampling"""
        n_samples = len(X)
        
        fig, axes = plt.subplots(1, self.n_trees, figsize=(15, 3))
        
        for i, ax in enumerate(axes):
            # Bootstrap sample
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            unique_samples = len(np.unique(bootstrap_idx))
            
            # Visualize
            ax.hist(bootstrap_idx, bins=20, alpha=0.7)
            ax.set_title(f'Tree {i+1}\n{unique_samples}/{n_samples} unique')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Frequency')
        
        plt.suptitle('Bootstrap Sampling for Random Forest', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def demonstrate_feature_randomness(self, n_features=10):
        """Show random feature selection"""
        max_features = int(np.sqrt(n_features))
        
        print("RANDOM FEATURE SELECTION")
        print("="*50)
        print(f"Total features: {n_features}")
        print(f"Max features per split: {max_features}")
        print("\nFeature subsets for each tree:")
        
        for i in range(self.n_trees):
            selected = np.random.choice(n_features, max_features, replace=False)
            print(f"Tree {i+1}: Features {sorted(selected)}")

# Demonstrate
demo = RandomForestDemonstration(n_trees=5)

# Bootstrap sampling
X_demo = np.random.randn(100, 10)
y_demo = np.random.randint(0, 2, 100)
demo.demonstrate_bootstrap(X_demo, y_demo)

# Feature randomness
demo.demonstrate_feature_randomness(n_features=20)
```

### Building a Random Forest

```python
# Comprehensive Random Forest example
from sklearn.datasets import load_wine

# Load data
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Build Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Evaluate
train_score = rf.score(X_train, y_train)
test_score = rf.score(X_test, y_test)
oob_score = rf.oob_score_

print("RANDOM FOREST RESULTS")
print("="*50)
print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")
print(f"OOB score: {oob_score:.3f}")

# Confusion matrix
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### Random Forest Convergence

```python
# Show how performance improves with more trees
n_estimators_range = [1, 5, 10, 20, 50, 100, 200, 500]
train_scores = []
test_scores = []
oob_scores = []

for n_estimators in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                               oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
    oob_scores.append(rf.oob_score_)

# Visualize convergence
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'b-', label='Training', marker='o')
plt.plot(n_estimators_range, test_scores, 'r-', label='Test', marker='s')
plt.plot(n_estimators_range, oob_scores, 'g-', label='OOB', marker='^')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest Performance vs Number of Trees')
plt.legend()
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.show()

print("Performance convergence:")
for n, train, test, oob in zip(n_estimators_range, train_scores, 
                               test_scores, oob_scores):
    print(f"Trees: {n:3d} | Train: {train:.3f} | Test: {test:.3f} | OOB: {oob:.3f}")
```

---

## Feature Importance

### Different Importance Measures

```python
from sklearn.inspection import permutation_importance

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 1. Mean Decrease in Impurity (MDI)
mdi_importance = rf.feature_importances_

# 2. Permutation Importance
perm_importance = permutation_importance(rf, X_test, y_test, 
                                        n_repeats=10, random_state=42)

# 3. Drop Column Importance
drop_importance = []
baseline_score = rf.score(X_test, y_test)

for i in range(X_test.shape[1]):
    X_test_dropped = np.delete(X_test, i, axis=1)
    X_train_dropped = np.delete(X_train, i, axis=1)
    
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_train_dropped, y_train)
    score = rf_temp.score(X_test_dropped, y_test)
    
    importance = baseline_score - score
    drop_importance.append(importance)

# Compare importance measures
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# MDI
ax = axes[0]
indices = np.argsort(mdi_importance)[::-1]
ax.barh(range(len(mdi_importance)), mdi_importance[indices])
ax.set_yticks(range(len(mdi_importance)))
ax.set_yticklabels([wine.feature_names[i] for i in indices])
ax.set_xlabel('Importance')
ax.set_title('Mean Decrease in Impurity')

# Permutation
ax = axes[1]
indices = np.argsort(perm_importance.importances_mean)[::-1]
ax.barh(range(len(indices)), perm_importance.importances_mean[indices])
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([wine.feature_names[i] for i in indices])
ax.set_xlabel('Importance')
ax.set_title('Permutation Importance')

# Drop column
ax = axes[2]
indices = np.argsort(drop_importance)[::-1]
ax.barh(range(len(drop_importance)), np.array(drop_importance)[indices])
ax.set_yticks(range(len(drop_importance)))
ax.set_yticklabels([wine.feature_names[i] for i in indices])
ax.set_xlabel('Accuracy Drop')
ax.set_title('Drop Column Importance')

plt.tight_layout()
plt.show()
```

### Feature Importance Stability

```python
# Check stability of feature importance across multiple runs
n_runs = 30
importance_runs = []

for run in range(n_runs):
    rf = RandomForestClassifier(n_estimators=100, random_state=run)
    rf.fit(X_train, y_train)
    importance_runs.append(rf.feature_importances_)

importance_runs = np.array(importance_runs)

# Calculate statistics
mean_importance = importance_runs.mean(axis=0)
std_importance = importance_runs.std(axis=0)

# Visualize stability
fig, ax = plt.subplots(figsize=(12, 6))

indices = np.argsort(mean_importance)[::-1]
x = np.arange(len(mean_importance))

ax.bar(x, mean_importance[indices], yerr=std_importance[indices], 
       capsize=5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([wine.feature_names[i] for i in indices], 
                   rotation=45, ha='right')
ax.set_xlabel('Feature')
ax.set_ylabel('Importance')
ax.set_title(f'Feature Importance Stability (n={n_runs} runs)')

plt.tight_layout()
plt.show()

# Print stability metrics
print("FEATURE IMPORTANCE STABILITY")
print("="*50)
for i, feature in enumerate(wine.feature_names):
    cv = std_importance[i] / mean_importance[i] if mean_importance[i] > 0 else 0
    print(f"{feature:20} Mean: {mean_importance[i]:.3f} ± {std_importance[i]:.3f} (CV: {cv:.2f})")
```

---

## Hyperparameter Tuning

### Grid Search for Random Forests

```python
# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Calculate total combinations
total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"Total parameter combinations: {total_combinations}")

# Use RandomizedSearchCV for efficiency
from sklearn.model_selection import RandomizedSearchCV

# Random search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=50,  # Sample 50 combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\nBEST PARAMETERS")
print("="*50)
for param, value in random_search.best_params_.items():
    print(f"{param:20} {value}")

print(f"\nBest CV Score: {random_search.best_score_:.3f}")

# Test set performance
best_rf = random_search.best_estimator_
test_score = best_rf.score(X_test, y_test)
print(f"Test Score: {test_score:.3f}")
```

### Hyperparameter Impact Analysis

```python
# Analyze impact of key hyperparameters
def analyze_hyperparameter(param_name, param_values, X_train, y_train, X_test, y_test):
    """Analyze the impact of a single hyperparameter"""
    
    train_scores = []
    test_scores = []
    oob_scores = []
    
    for value in param_values:
        params = {param_name: value, 'oob_score': True, 'random_state': 42}
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        
        train_scores.append(rf.score(X_train, y_train))
        test_scores.append(rf.score(X_test, y_test))
        oob_scores.append(rf.oob_score_)
    
    return train_scores, test_scores, oob_scores

# Analyze different hyperparameters
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# n_estimators
ax = axes[0, 0]
param_values = [10, 50, 100, 200, 500]
train, test, oob = analyze_hyperparameter('n_estimators', param_values, 
                                         X_train, y_train, X_test, y_test)
ax.plot(param_values, train, 'b-', label='Train', marker='o')
ax.plot(param_values, test, 'r-', label='Test', marker='s')
ax.plot(param_values, oob, 'g-', label='OOB', marker='^')
ax.set_xlabel('n_estimators')
ax.set_ylabel('Accuracy')
ax.set_title('Impact of Number of Trees')
ax.legend()
ax.grid(True, alpha=0.3)

# max_depth
ax = axes[0, 1]
param_values = [2, 5, 10, 20, None]
param_labels = [str(v) if v else 'None' for v in param_values]
train, test, oob = analyze_hyperparameter('max_depth', param_values,
                                         X_train, y_train, X_test, y_test)
x = range(len(param_values))
ax.plot(x, train, 'b-', label='Train', marker='o')
ax.plot(x, test, 'r-', label='Test', marker='s')
ax.plot(x, oob, 'g-', label='OOB', marker='^')
ax.set_xticks(x)
ax.set_xticklabels(param_labels)
ax.set_xlabel('max_depth')
ax.set_ylabel('Accuracy')
ax.set_title('Impact of Maximum Depth')
ax.legend()
ax.grid(True, alpha=0.3)

# min_samples_split
ax = axes[1, 0]
param_values = [2, 5, 10, 20, 50]
train, test, oob = analyze_hyperparameter('min_samples_split', param_values,
                                         X_train, y_train, X_test, y_test)
ax.plot(param_values, train, 'b-', label='Train', marker='o')
ax.plot(param_values, test, 'r-', label='Test', marker='s')
ax.plot(param_values, oob, 'g-', label='OOB', marker='^')
ax.set_xlabel('min_samples_split')
ax.set_ylabel('Accuracy')
ax.set_title('Impact of Minimum Samples Split')
ax.legend()
ax.grid(True, alpha=0.3)

# max_features
ax = axes[1, 1]
n_features = X_train.shape[1]
param_values = [1, int(np.sqrt(n_features)), int(n_features/2), n_features]
train, test, oob = analyze_hyperparameter('max_features', param_values,
                                         X_train, y_train, X_test, y_test)
ax.plot(param_values, train, 'b-', label='Train', marker='o')
ax.plot(param_values, test, 'r-', label='Test', marker='s')
ax.plot(param_values, oob, 'g-', label='OOB', marker='^')
ax.set_xlabel('max_features')
ax.set_ylabel('Accuracy')
ax.set_title('Impact of Maximum Features')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Hyperparameter Impact Analysis', fontsize=16)
plt.tight_layout()
plt.show()
```

---

## Advanced Techniques

### Extremely Randomized Trees

```python
from sklearn.ensemble import ExtraTreesClassifier

# Compare Random Forest vs Extra Trees
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Timing
    import time
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    results.append({
        'Model': name,
        'Train Score': train_score,
        'Test Score': test_score,
        'Training Time': train_time
    })

results_df = pd.DataFrame(results)
print("RANDOM FOREST vs EXTRA TREES")
print("="*60)
print(results_df.to_string(index=False))
```

### Feature Selection with Random Forests

```python
from sklearn.feature_selection import SelectFromModel

# Use Random Forest for feature selection
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train, y_train)

# Select features
selector = SelectFromModel(rf_selector, threshold='median')
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

print("FEATURE SELECTION WITH RANDOM FOREST")
print("="*50)
print(f"Original features: {X_train.shape[1]}")
print(f"Selected features: {X_train_selected.shape[1]}")

# Compare performance
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train, y_train)
score_all = rf_all.score(X_test, y_test)

rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)
score_selected = rf_selected.score(X_test_selected, y_test)

print(f"\nAccuracy with all features: {score_all:.3f}")
print(f"Accuracy with selected features: {score_selected:.3f}")

# Show selected features
selected_features = selector.get_support()
selected_names = [wine.feature_names[i] for i, selected in enumerate(selected_features) if selected]
print(f"\nSelected features: {selected_names}")
```

---

## Real-World Applications

### Project: Credit Risk Assessment

```python
# Simulate credit risk dataset
np.random.seed(42)
n_samples = 5000

# Generate features
credit_data = pd.DataFrame({
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.lognormal(10.5, 0.5, n_samples),
    'employment_years': np.random.randint(0, 40, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'num_accounts': np.random.randint(1, 20, n_samples),
    'num_late_payments': np.random.poisson(1, n_samples),
    'debt_ratio': np.random.beta(2, 5, n_samples),
    'has_mortgage': np.random.choice([0, 1], n_samples),
    'has_car_loan': np.random.choice([0, 1], n_samples)
})

# Generate target based on features (simplified model)
default_probability = (
    (credit_data['credit_score'] < 600).astype(float) * 0.4 +
    (credit_data['debt_ratio'] > 0.5).astype(float) * 0.3 +
    (credit_data['num_late_payments'] > 3).astype(float) * 0.2 +
    np.random.random(n_samples) * 0.1
)
credit_data['default'] = (default_probability > 0.5).astype(int)

print("CREDIT RISK ASSESSMENT PROJECT")
print("="*50)
print(f"Dataset shape: {credit_data.shape}")
print(f"Default rate: {credit_data['default'].mean():.2%}")
print("\nFeature statistics:")
print(credit_data.describe())

# Prepare data
X = credit_data.drop('default', axis=1)
y = credit_data['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build and optimize Random Forest
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_credit = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',  # Handle imbalanced classes
    random_state=42
)

rf_credit.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_credit.predict(X_test_scaled)
y_proba = rf_credit.predict_proba(X_test_scaled)[:, 1]

# Evaluation
from sklearn.metrics import classification_report, roc_auc_score

print("\nMODEL PERFORMANCE")
print("="*50)
print(classification_report(y_test, y_pred, 
                           target_names=['No Default', 'Default']))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {roc_auc:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_credit.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFEATURE IMPORTANCE")
print("="*50)
print(feature_importance.to_string(index=False))

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROC Curve
from sklearn.metrics import roc_curve
ax = axes[0, 0]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()

# Feature Importance
ax = axes[0, 1]
ax.barh(range(len(feature_importance)), feature_importance['importance'])
ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['feature'])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance')

# Confusion Matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')

# Probability Distribution
ax = axes[1, 1]
ax.hist(y_proba[y_test == 0], bins=30, alpha=0.5, label='No Default', density=True)
ax.hist(y_proba[y_test == 1], bins=30, alpha=0.5, label='Default', density=True)
ax.set_xlabel('Predicted Probability of Default')
ax.set_ylabel('Density')
ax.set_title('Probability Distribution by Class')
ax.legend()

plt.tight_layout()
plt.show()

# Business metrics
threshold = 0.3  # Business-defined threshold
y_pred_business = (y_proba >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_business).ravel()

print("\nBUSINESS METRICS (Threshold = 0.3)")
print("="*50)
print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.3f}")
print(f"Precision: {tp / (tp + fp):.3f}")
print(f"Recall: {tp / (tp + fn):.3f}")
print(f"False Positive Rate: {fp / (fp + tn):.3f}")
print(f"False Negative Rate: {fn / (fn + tp):.3f}")
```

---

## Summary and Best Practices

### Key Takeaways

1. **Decision Trees**:
   - Interpretable and require no preprocessing
   - Prone to overfitting without pruning
   - Use pruning parameters (max_depth, min_samples_split)

2. **Random Forests**:
   - Ensemble of decision trees with bagging
   - Reduces overfitting through averaging
   - Robust to outliers and noise

3. **Feature Importance**:
   - Multiple methods available (MDI, permutation)
   - Use for feature selection and interpretation
   - Check stability across runs

4. **Hyperparameter Tuning**:
   - n_estimators: More trees = better (but diminishing returns)
   - max_depth: Control overfitting
   - max_features: sqrt for classification, n/3 for regression

5. **Best Practices**:
   - Always use cross-validation
   - Check OOB score for model selection
   - Consider computational resources
   - Use feature importance for insights

### When to Use Random Forests

```python
# Decision guide
use_cases = {
    "✅ Good for Random Forests": [
        "Tabular data with mixed types",
        "Non-linear relationships",
        "Feature importance needed",
        "Robust predictions required",
        "Moderate-sized datasets"
    ],
    "❌ Consider Alternatives": [
        "Very high-dimensional data (use regularized linear models)",
        "Sequential/time series data (use RNNs/ARIMA)",
        "Image data (use CNNs)",
        "Text data (use NLP models)",
        "Need probability calibration (use calibrated classifiers)"
    ]
}

for category, items in use_cases.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  • {item}")
```

### Performance Tips

```python
# Optimization strategies
optimization_tips = """
RANDOM FOREST OPTIMIZATION TIPS
================================

1. Start Simple:
   - Begin with default parameters
   - Use OOB score for quick evaluation
   
2. Computational Efficiency:
   - Use n_jobs=-1 for parallel processing
   - Consider ExtraTreesClassifier for speed
   - Reduce max_depth for faster training
   
3. Memory Management:
   - Use max_samples to limit bootstrap size
   - Consider sparse matrices for high-dimensional data
   
4. Model Size:
   - Use min_samples_leaf to reduce tree size
   - Compress models with joblib
   
5. Production Deployment:
   - Monitor feature drift
   - Retrain periodically
   - Use model versioning
"""

print(optimization_tips)
```

### Final Checklist

```python
checklist = """
RANDOM FOREST IMPLEMENTATION CHECKLIST
======================================
□ Data preparation
  □ Handle missing values
  □ Encode categorical variables
  □ Scale features (optional for RF)
  
□ Model training
  □ Split data (train/validation/test)
  □ Start with default parameters
  □ Check OOB score
  
□ Hyperparameter tuning
  □ Use cross-validation
  □ Try RandomizedSearchCV first
  □ Focus on n_estimators, max_depth, max_features
  
□ Evaluation
  □ Check train vs test performance
  □ Analyze feature importance
  □ Validate on holdout set
  
□ Deployment
  □ Save model and preprocessors
  □ Document feature requirements
  □ Set up monitoring
"""

print(checklist)
```

---

[← Back to Main](../README.md)