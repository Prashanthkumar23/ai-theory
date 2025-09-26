# Module 4: Supervised Learning Fundamentals 📈

[← Previous Module](03_intro_to_ml.md) | [Back to Main](../README.md) | [Next Module →](05_unsupervised_learning.md)

## 📋 Table of Contents
1. [Introduction to Supervised Learning](#introduction-to-supervised-learning)
2. [Linear Regression](#linear-regression)
3. [Logistic Regression](#logistic-regression)
4. [Support Vector Machines](#support-vector-machines)
5. [Gradient Descent](#gradient-descent)
6. [Regularization Techniques](#regularization-techniques)
7. [Model Evaluation](#model-evaluation)
8. [Cross-Validation Strategies](#cross-validation-strategies)
9. [Practical Implementation](#practical-implementation)
10. [Common Pitfalls & Solutions](#common-pitfalls--solutions)

---

## Introduction to Supervised Learning

Supervised learning is like learning with a teacher. You have input data (features) and correct answers (labels), and the goal is to learn a mapping function from inputs to outputs.

### Key Characteristics
- **Training with labeled data**: Each example has an input and known output
- **Goal**: Predict outputs for new, unseen inputs
- **Types**:
  - **Regression**: Predict continuous values (house prices, temperature)
  - **Classification**: Predict discrete categories (spam/not spam, digit recognition)

### The Learning Process
```
Training Data → Algorithm → Model → Predictions
     ↑                         ↓
   Labels                   Evaluation
```

---

## Linear Regression

### Concept in Simple Terms
Imagine drawing the best-fit line through a scatter plot of points. That's linear regression!

### Mathematical Foundation

#### Simple Linear Regression
For a single feature:
```
y = mx + b
```
Where:
- `y` = predicted value
- `m` = slope (weight)
- `x` = input feature
- `b` = y-intercept (bias)

#### Multiple Linear Regression
For multiple features:
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"Coefficients: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
plt.plot(X_test, y_pred, 'r-', label='Predicted', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Results')
plt.show()
```

### Implementing from Scratch

```python
class SimpleLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        """Fit using closed-form solution (Normal Equation)"""
        n = len(X)

        # Calculate means
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Calculate slope (w) and intercept (b)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean

    def predict(self, X):
        return self.w * X + self.b

    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
```

### Key Assumptions
1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Features are not highly correlated

---

## Logistic Regression

### Intuition
Despite the name, logistic regression is for classification! It predicts probabilities between 0 and 1.

### The Sigmoid Function
Transforms any value to range [0, 1]:
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### Mathematical Foundation
```
p(y=1|x) = σ(w^T x + b) = 1 / (1 + e^(-(w^T x + b)))
```

### Implementation Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate binary classification data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Predict
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Multiclass Classification
Using One-vs-Rest (OvR) or Softmax:

```python
# Multiclass example
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with multi_class
multi_log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
multi_log_reg.fit(X_train, y_train)

y_pred = multi_log_reg.predict(X_test)
print(f"Multiclass Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

---

## Support Vector Machines

### Intuition
Find the best separating hyperplane with maximum margin between classes.

### Key Concepts

#### Linear SVM
```python
from sklearn.svm import SVC
import numpy as np

# Create sample data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Train linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X, y)

# Visualize decision boundary
def plot_decision_boundary(model, X, y):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_decision_boundary(svm_linear, X, y)
```

#### Non-linear SVM with Kernels

```python
# RBF Kernel for non-linear data
from sklearn.datasets import make_moons

X_moon, y_moon = make_moons(n_samples=200, noise=0.15, random_state=42)

# Compare linear vs RBF kernel
svm_linear = SVC(kernel='linear', C=1.0)
svm_rbf = SVC(kernel='rbf', gamma='auto', C=1.0)

svm_linear.fit(X_moon, y_moon)
svm_rbf.fit(X_moon, y_moon)

print(f"Linear SVM Accuracy: {svm_linear.score(X_moon, y_moon):.2f}")
print(f"RBF SVM Accuracy: {svm_rbf.score(X_moon, y_moon):.2f}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Grid search for best parameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")
```

---

## Gradient Descent

### The Optimization Workhorse
Gradient descent is how most ML models learn - by iteratively moving towards the minimum of a loss function.

### Types of Gradient Descent

#### 1. Batch Gradient Descent
Updates using entire dataset:
```python
def batch_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])
    cost_history = []

    for i in range(n_iterations):
        # Compute predictions
        predictions = X.dot(theta)

        # Compute errors
        errors = predictions - y

        # Update parameters
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient

        # Store cost
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)

    return theta, cost_history
```

#### 2. Stochastic Gradient Descent (SGD)
Updates using one sample at a time:
```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])
    cost_history = []

    for iteration in range(n_iterations):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            prediction = xi.dot(theta)
            error = prediction - yi
            gradient = xi.T.dot(error)
            theta = theta - learning_rate * gradient

        # Calculate cost for monitoring
        cost = (1/(2*m)) * np.sum((X.dot(theta) - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

#### 3. Mini-batch Gradient Descent
Best of both worlds:
```python
def mini_batch_gradient_descent(X, y, batch_size=32, learning_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])
    cost_history = []

    for iteration in range(n_iterations):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Process mini-batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            predictions = X_batch.dot(theta)
            errors = predictions - y_batch
            gradient = (1/len(y_batch)) * X_batch.T.dot(errors)
            theta = theta - learning_rate * gradient

        # Calculate cost
        cost = (1/(2*m)) * np.sum((X.dot(theta) - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

### Learning Rate Scheduling

```python
def learning_rate_scheduler(initial_lr, iteration, decay_rate=0.95, decay_steps=100):
    """Exponential decay learning rate"""
    return initial_lr * (decay_rate ** (iteration / decay_steps))

# Example with scheduling
def gradient_descent_with_scheduling(X, y, initial_lr=0.1, n_iterations=1000):
    theta = np.zeros(X.shape[1])
    m = len(y)

    for i in range(n_iterations):
        lr = learning_rate_scheduler(initial_lr, i)
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - lr * gradient

    return theta
```

---

## Regularization Techniques

### Preventing Overfitting
Regularization adds a penalty term to prevent the model from fitting noise.

### L2 Regularization (Ridge)
Penalizes large weights:
```python
from sklearn.linear_model import Ridge

# Ridge Regression
ridge = Ridge(alpha=1.0)  # alpha is regularization strength
ridge.fit(X_train, y_train)

# Manual implementation
def ridge_regression(X, y, alpha=1.0):
    """Closed-form solution for Ridge Regression"""
    n_features = X.shape[1]
    identity = np.eye(n_features)
    identity[0, 0] = 0  # Don't regularize bias term

    # Ridge formula: (X^T X + alpha*I)^(-1) X^T y
    theta = np.linalg.inv(X.T.dot(X) + alpha * identity).dot(X.T).dot(y)
    return theta
```

### L1 Regularization (Lasso)
Creates sparse models by pushing weights to zero:
```python
from sklearn.linear_model import Lasso

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Feature selection property
print(f"Number of zero coefficients: {np.sum(lasso.coef_ == 0)}")
print(f"Selected features: {np.where(lasso.coef_ != 0)[0]}")
```

### Elastic Net
Combines L1 and L2:
```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)  # l1_ratio balances L1 vs L2
elastic.fit(X_train, y_train)
```

### Comparing Regularization Methods

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create pipelines with different regularization
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    results[name] = score
    print(f"{name}: R² = {score:.3f}")
```

---

## Model Evaluation

### Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_regression_model(y_true, y_pred):
    """Comprehensive regression evaluation"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mape': mape}
```

### Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns

def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """Comprehensive classification evaluation"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")

    if y_prob is not None and len(np.unique(y_true)) == 2:
        # Binary classification - compute AUC
        auc = roc_auc_score(y_true, y_prob[:, 1])
        print(f"AUC-ROC: {auc:.3f}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
```

---

## Cross-Validation Strategies

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold, cross_val_score

def perform_kfold_cv(model, X, y, k=5):
    """Perform k-fold cross-validation"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

    # Convert to positive RMSE
    rmse_scores = np.sqrt(-scores)

    print(f"K-Fold CV Results (k={k}):")
    print(f"RMSE scores: {rmse_scores}")
    print(f"Mean RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std() * 2:.3f})")

    return rmse_scores
```

### Stratified K-Fold for Classification

```python
from sklearn.model_selection import StratifiedKFold

def perform_stratified_cv(model, X, y, k=5):
    """Stratified K-Fold for balanced class distribution"""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    print(f"Stratified K-Fold CV Results (k={k}):")
    print(f"Accuracy scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

    return scores
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(model, X, y, n_splits=5):
    """Time series specific cross-validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        print(f"Fold {i+1}: Train size={len(train_idx)}, Test size={len(test_idx)}, Score={score:.3f}")

    print(f"Mean Score: {np.mean(scores):.3f}")
    return scores
```

---

## Practical Implementation

### Complete Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV

def create_regression_pipeline():
    """Create a complete ML pipeline with preprocessing and model"""

    # Define pipeline steps
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_regression)),
        ('model', Ridge())
    ])

    # Define parameter grid for GridSearch
    param_grid = {
        'poly__degree': [1, 2, 3],
        'feature_selection__k': [5, 10, 15, 20],
        'model__alpha': [0.1, 1.0, 10.0]
    }

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                              scoring='neg_mean_squared_error', n_jobs=-1)

    return grid_search

# Usage example
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train pipeline
pipeline = create_regression_pipeline()
pipeline.fit(X_train, y_train)

print(f"Best parameters: {pipeline.best_params_}")
print(f"Test score: {pipeline.score(X_test, y_test):.3f}")
```

### Real-world Example: House Price Prediction

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Simulated housing data
def create_housing_data():
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({
        'size': np.random.normal(1500, 500, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'neighborhood': np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples),
        'garage': np.random.choice([0, 1, 2], n_samples)
    })

    # Create price based on features (with some noise)
    price = (
        data['size'] * 100 +
        data['bedrooms'] * 10000 +
        data['bathrooms'] * 15000 -
        data['age'] * 1000 +
        (data['neighborhood'] == 'Downtown').astype(int) * 50000 +
        data['garage'] * 20000 +
        np.random.normal(0, 20000, n_samples)
    )

    data['price'] = price
    return data

# Create and prepare data
housing_data = create_housing_data()
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Identify numeric and categorical columns
numeric_features = ['size', 'bedrooms', 'bathrooms', 'age', 'garage']
categorical_features = ['neighborhood']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Create full pipeline
housing_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
housing_pipeline.fit(X_train, y_train)

# Predictions
y_pred = housing_pipeline.predict(X_test)

# Evaluate
print("House Price Prediction Results:")
evaluate_regression_model(y_test, y_pred)

# Feature importance (for linear models)
model_coef = housing_pipeline.named_steps['model'].coef_
feature_names = (numeric_features +
                ['neighborhood_Rural', 'neighborhood_Suburb'])
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(model_coef)
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
```

---

## Common Pitfalls & Solutions

### 1. Data Leakage
**Problem**: Information from test set leaks into training
```python
# WRONG: Scaling before splitting
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fits on entire dataset!
X_train, X_test = train_test_split(X_scaled)

# CORRECT: Scale after splitting
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit!
```

### 2. Ignoring Class Imbalance
**Solution**: Use balanced metrics and sampling
```python
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

# Option 1: Class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train)
model = LogisticRegression(class_weight='balanced')

# Option 2: SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

### 3. Not Checking Assumptions
```python
# Check for multicollinearity
def check_multicollinearity(X, threshold=5):
    """Calculate Variance Inflation Factor (VIF)"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(len(X.columns))]

    print("Variance Inflation Factors:")
    print(vif_data)
    print(f"\nFeatures with VIF > {threshold} may have multicollinearity issues")
    return vif_data
```

### 4. Overfitting to Training Data
```python
# Learning curves to diagnose overfitting
from sklearn.model_selection import learning_curve

def plot_learning_curves(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10))

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.title('Learning Curves')
    plt.grid(True)
    plt.show()
```

### 5. Wrong Metric for the Problem
```python
# Choose appropriate metrics
def select_metric(problem_type, data_characteristics):
    """Guide for metric selection"""

    if problem_type == 'regression':
        if 'outliers' in data_characteristics:
            return 'mae'  # More robust to outliers
        elif 'percentage_errors' in data_characteristics:
            return 'mape'
        else:
            return 'rmse'

    elif problem_type == 'classification':
        if 'imbalanced' in data_characteristics:
            return 'f1_weighted'  # or 'roc_auc' for binary
        elif 'cost_sensitive' in data_characteristics:
            return 'custom_cost_function'
        else:
            return 'accuracy'
```

---

## Practice Exercises

### Exercise 1: Implement Gradient Descent Variants
```python
# TODO: Implement and compare BGD, SGD, and Mini-batch GD
# on the same dataset and plot convergence curves
```

### Exercise 2: Regularization Comparison
```python
# TODO: Create synthetic data with many features
# Compare Linear, Ridge, Lasso, and ElasticNet
# Plot coefficient values for each method
```

### Exercise 3: Build a Complete Classification Pipeline
```python
# TODO: Use the Iris dataset
# 1. Implement data preprocessing
# 2. Compare multiple classifiers
# 3. Perform hyperparameter tuning
# 4. Evaluate with appropriate metrics
# 5. Visualize decision boundaries
```

### Exercise 4: Cross-Validation Implementation
```python
# TODO: Implement k-fold CV from scratch
# Compare with sklearn's implementation
# Test on both regression and classification problems
```

---

## Summary and Key Takeaways

### Supervised Learning Essentials
1. **Linear Regression**: Foundation for understanding relationships
2. **Logistic Regression**: Probability-based classification
3. **SVM**: Maximum margin classification with kernel trick
4. **Gradient Descent**: The engine of optimization
5. **Regularization**: Preventing overfitting through penalties
6. **Evaluation**: Choose metrics that align with your goals
7. **Cross-Validation**: Robust performance estimation

### Best Practices Checklist
- [ ] Always split data before preprocessing
- [ ] Scale/normalize features for distance-based algorithms
- [ ] Check for class imbalance in classification
- [ ] Use cross-validation for model selection
- [ ] Monitor both training and validation performance
- [ ] Consider regularization for high-dimensional data
- [ ] Choose metrics appropriate for your problem
- [ ] Validate assumptions before applying algorithms
- [ ] Use pipelines for reproducible workflows
- [ ] Document preprocessing steps and model choices

### When to Use What?

| Algorithm | Use When | Avoid When |
|-----------|----------|------------|
| Linear Regression | Linear relationships, interpretability needed | Non-linear patterns, outliers present |
| Logistic Regression | Binary/multiclass classification, probability estimates | Non-linear boundaries, very high dimensions |
| SVM | Non-linear patterns (with kernels), high-dimensional data | Large datasets (slow), probability estimates needed |
| Ridge | Many features, multicollinearity | Feature selection needed |
| Lasso | Feature selection needed, sparse solutions | Groups of correlated features |
| ElasticNet | Balance between Ridge and Lasso | Simple linear relationships |

---

## Additional Resources

### Libraries to Master
- **scikit-learn**: Main ML library
- **statsmodels**: Statistical modeling
- **XGBoost/LightGBM**: Advanced gradient boosting (next modules)
- **optuna**: Hyperparameter optimization

### Recommended Projects
1. Build a spam classifier from scratch
2. Predict house prices with real estate data
3. Create a credit risk model
4. Implement all algorithms without sklearn
5. Build an automated ML pipeline

### Next Steps
- Module 5: Explore unsupervised learning techniques
- Module 6: Dive into neural networks and deep learning
- Module 7: Master scikit-learn's advanced features
- Module 8: Learn ensemble methods

---

[**Continue to Module 5: Unsupervised Learning →**](05_unsupervised_learning.md)