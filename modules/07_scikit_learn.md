# Module 7: Scikit-learn - A Practical Introduction 🛠️

[← Previous Module](06_neural_networks.md) | [Back to Main](../README.md) | [Next Module →](08_decision_trees_random_forests.md)

## 📋 Table of Contents
1. [Introduction to Scikit-learn](#introduction-to-scikit-learn)
2. [Installation and Setup](#installation-and-setup)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training and Evaluation](#model-training-and-evaluation)
5. [Pipeline Creation](#pipeline-creation)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Model Persistence](#model-persistence)
8. [Advanced Features](#advanced-features)
9. [Common Algorithms in Scikit-learn](#common-algorithms-in-scikit-learn)
10. [Best Practices](#best-practices)
11. [Real-World Projects](#real-world-projects)
12. [Summary and Resources](#summary-and-resources)

---

## Introduction to Scikit-learn

Scikit-learn is the most popular machine learning library for Python, providing:
- Simple and efficient tools for data mining and data analysis
- Consistent API across all algorithms
- Built on NumPy, SciPy, and matplotlib
- Open source, commercially usable (BSD license)

![Scikit-learn Overview](https://scikit-learn.org/stable/_static/ml_map.png)

### Why Scikit-learn?

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import __version__

print(f"Scikit-learn version: {__version__}")

# Key advantages of scikit-learn
advantages = {
    "Consistent API": "fit(), predict(), transform() across all models",
    "Comprehensive": "Classification, Regression, Clustering, Dimensionality reduction",
    "Well-documented": "Extensive documentation with examples",
    "Efficient": "Optimized implementations in Cython",
    "Integration": "Works seamlessly with NumPy, Pandas, matplotlib"
}

for key, value in advantages.items():
    print(f"✓ {key}: {value}")
```

---

## Installation and Setup

### Installing Scikit-learn

```bash
# Using pip
pip install scikit-learn

# Using conda
conda install scikit-learn

# Install with all optional dependencies
pip install scikit-learn[alldeps]

# For development (latest features)
pip install --pre scikit-learn
```

### Checking Your Installation

```python
# Verify installation
import sklearn
from sklearn import datasets, preprocessing, model_selection, metrics

# Check version and configuration
sklearn.show_versions()

# Load a sample dataset to verify everything works
iris = datasets.load_iris()
print(f"Dataset shape: {iris.data.shape}")
print(f"Target names: {iris.target_names}")
```

---

## Data Preprocessing

### 1. Loading Data

```python
# Different ways to load data in scikit-learn

# 1. Built-in datasets
from sklearn import datasets

# Classification datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
wine = datasets.load_wine()

# Regression datasets
boston = datasets.fetch_california_housing()
diabetes = datasets.load_diabetes()

# 2. Generate synthetic data
from sklearn.datasets import make_classification, make_regression, make_blobs

# Classification data
X_class, y_class = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

# Regression data
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    noise=0.1,
    random_state=42
)

# Clustering data
X_clust, y_clust = make_blobs(
    n_samples=1000,
    centers=4,
    n_features=2,
    random_state=42
)

# 3. Load from external files
# From CSV
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

print(f"Classification data: {X_class.shape}")
print(f"Regression data: {X_reg.shape}")
print(f"Clustering data: {X_clust.shape}")
```

### 2. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

# Generate sample data
np.random.seed(42)
X = np.array([[1, 2000],
              [2, 3000],
              [3, 4000],
              [4, 5000],
              [5, 6000]])

# Different scaling methods
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'Normalizer': Normalizer()
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, scaler) in zip(axes.flat, scalers.items()):
    X_scaled = scaler.fit_transform(X)
    
    # Visualize
    ax.scatter(X[:, 0], X[:, 1], label='Original', s=100, alpha=0.5)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], label='Scaled', s=100, alpha=0.5)
    ax.set_title(f'{name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.text(0.02, 0.98, 
           f'Original: μ={X.mean():.1f}, σ={X.std():.1f}\n'
           f'Scaled: μ={X_scaled.mean():.3f}, σ={X_scaled.std():.3f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
```

### 3. Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Sample categorical data
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue', 'red'],
    'size': ['S', 'M', 'L', 'XL', 'M'],
    'quality': ['good', 'bad', 'excellent', 'good', 'bad']
})

print("Original Data:")
print(data)
print("\n" + "="*50 + "\n")

# 1. Label Encoding (for ordinal features or targets)
le = LabelEncoder()
data['quality_encoded'] = le.fit_transform(data['quality'])
print("Label Encoding (quality):")
print(data[['quality', 'quality_encoded']])
print(f"Classes: {le.classes_}")
print("\n" + "="*50 + "\n")

# 2. Ordinal Encoding (when order matters)
oe = OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']])
data['size_encoded'] = oe.fit_transform(data[['size']])
print("Ordinal Encoding (size with order):")
print(data[['size', 'size_encoded']])
print("\n" + "="*50 + "\n")

# 3. One-Hot Encoding (for nominal features)
ohe = OneHotEncoder(sparse=False)
color_encoded = ohe.fit_transform(data[['color']])
color_df = pd.DataFrame(color_encoded, columns=ohe.get_feature_names_out(['color']))
print("One-Hot Encoding (color):")
print(pd.concat([data[['color']], color_df], axis=1))
```

### 4. Handling Missing Data

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Create data with missing values
X_missing = np.array([[1, 2, np.nan],
                      [3, np.nan, 6],
                      [7, 8, 9],
                      [np.nan, 10, 11],
                      [12, 13, 14]])

print("Original data with missing values:")
print(X_missing)
print("\n" + "="*50 + "\n")

# Different imputation strategies
imputers = {
    'Mean Imputation': SimpleImputer(strategy='mean'),
    'Median Imputation': SimpleImputer(strategy='median'),
    'Most Frequent': SimpleImputer(strategy='most_frequent'),
    'KNN Imputation': KNNImputer(n_neighbors=2),
    'Iterative Imputation': IterativeImputer(random_state=42)
}

for name, imputer in imputers.items():
    X_imputed = imputer.fit_transform(X_missing)
    print(f"{name}:")
    print(X_imputed)
    print()
```

### 5. Feature Selection

```python
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, f_classif, 
    mutual_info_classif, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier

# Generate data
X, y = make_classification(n_samples=200, n_features=20, 
                          n_informative=10, n_redundant=10,
                          random_state=42)

feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

# 1. Univariate Selection
selector_univariate = SelectKBest(f_classif, k=10)
X_univariate = selector_univariate.fit_transform(X, y)
scores = selector_univariate.scores_

# 2. Recursive Feature Elimination
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector_rfe = RFE(estimator, n_features_to_select=10)
X_rfe = selector_rfe.fit_transform(X, y)

# 3. Tree-based Selection
selector_tree = SelectFromModel(estimator, threshold='median')
selector_tree.fit(X, y)
X_tree = selector_tree.transform(X)

# Visualize feature importance
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Univariate scores
ax = axes[0]
indices = np.argsort(scores)[::-1][:10]
ax.bar(range(10), scores[indices])
ax.set_title('Univariate Feature Selection')
ax.set_xlabel('Feature Index')
ax.set_ylabel('F-Score')

# RFE ranking
ax = axes[1]
ranking = selector_rfe.ranking_
ax.bar(range(len(ranking)), 1/ranking)
ax.set_title('RFE Feature Ranking')
ax.set_xlabel('Feature Index')
ax.set_ylabel('Importance')

# Tree-based importance
ax = axes[2]
importances = estimator.feature_importances_
indices = np.argsort(importances)[::-1][:10]
ax.bar(range(10), importances[indices])
ax.set_title('Tree-based Feature Importance')
ax.set_xlabel('Feature Index')
ax.set_ylabel('Importance')

plt.tight_layout()
plt.show()

print(f"Original features: {X.shape[1]}")
print(f"After univariate selection: {X_univariate.shape[1]}")
print(f"After RFE: {X_rfe.shape[1]}")
print(f"After tree-based selection: {X_tree.shape[1]}")
```

---

## Model Training and Evaluation

### Complete Training Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load and prepare data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# For binary classification (simplify to 2 classes)
binary_mask = y != 2
X_binary = X[binary_mask]
y_binary = y[binary_mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

print("COMPLETE MODEL TRAINING WORKFLOW")
print("="*50)

# Step 1: Initialize model
model = LogisticRegression(random_state=42)
print("Step 1: Model initialized")

# Step 2: Train model
model.fit(X_train, y_train)
print("Step 2: Model trained")

# Step 3: Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
print("Step 3: Predictions made")

# Step 4: Evaluate model
accuracy = model.score(X_test, y_test)
print(f"Step 4: Model evaluated - Accuracy: {accuracy:.3f}")

# Detailed evaluation
print("\nDETAILED EVALUATION:")
print("="*50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names[:2]))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ROC AUC for binary classification
roc_auc = roc_auc_score(y_test, y_proba[:, 1])
print(f"\nROC AUC Score: {roc_auc:.3f}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Confusion Matrix
ax = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# ROC Curve
from sklearn.metrics import roc_curve
ax = axes[1]
fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()

# Feature Coefficients
ax = axes[2]
coefficients = model.coef_[0]
feature_names = iris.feature_names
ax.barh(range(len(coefficients)), coefficients)
ax.set_yticks(range(len(coefficients)))
ax.set_yticklabels(feature_names)
ax.set_xlabel('Coefficient Value')
ax.set_title('Feature Importance (Logistic Regression)')

plt.tight_layout()
plt.show()
```

---

## Pipeline Creation

### Building Robust ML Pipelines

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Create sample data with mixed types
data = pd.DataFrame({
    'age': [25, 30, 35, np.nan, 45],
    'salary': [50000, 60000, np.nan, 80000, 90000],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
    'education': ['BS', 'MS', 'PhD', 'BS', 'MS'],
    'target': [0, 1, 1, 0, 1]
})

X = data.drop('target', axis=1)
y = data['target']

print("Sample Data:")
print(data)
print("\n" + "="*50 + "\n")

# Define column groups
numeric_features = ['age', 'salary']
categorical_features = ['city', 'education']

# Create preprocessing pipelines for each type
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create full pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
full_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = full_pipeline.predict(X_test)

print("Pipeline Steps:")
for step_name, step in full_pipeline.steps:
    print(f"  - {step_name}: {type(step).__name__}")

print(f"\nPipeline Score: {full_pipeline.score(X_test, y_test):.3f}")

# Visualize pipeline
from sklearn import set_config
set_config(display='diagram')
display(full_pipeline)  # In Jupyter notebook
```

### Advanced Pipeline with Feature Union

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

# Create feature extraction pipeline
feature_extraction = FeatureUnion([
    ('pca', PCA(n_components=2)),
    ('select_best', SelectKBest(k=2))
])

# Complex pipeline
complex_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('features', feature_extraction),
    ('classifier', LogisticRegression())
])

# Use with iris dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

complex_pipeline.fit(X_train, y_train)
score = complex_pipeline.score(X_test, y_test)

print(f"Complex Pipeline Score: {score:.3f}")

# Access intermediate results
X_scaled = complex_pipeline.named_steps['scaler'].transform(X_train)
X_features = complex_pipeline.named_steps['features'].transform(X_scaled)

print(f"Original shape: {X_train.shape}")
print(f"After scaling: {X_scaled.shape}")
print(f"After feature extraction: {X_features.shape}")
```

---

## Hyperparameter Tuning

### 1. Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Prepare data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'linear', 'poly']
}

# Create grid search
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X, y)

print("GRID SEARCH RESULTS")
print("="*50)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Analyze results
results_df = pd.DataFrame(grid_search.cv_results_)
top_10 = results_df.nlargest(10, 'mean_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
print("\nTop 10 parameter combinations:")
print(top_10)

# Visualize grid search results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap for RBF kernel
rbf_results = results_df[results_df['param_kernel'] == 'rbf']
pivot_table = rbf_results.pivot_table(
    values='mean_test_score',
    index='param_C',
    columns='param_gamma'
)

ax = axes[0]
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
ax.set_title('Grid Search Heatmap (RBF Kernel)')
ax.set_xlabel('Gamma')
ax.set_ylabel('C')

# Parameter importance
ax = axes[1]
param_importance = results_df.groupby('param_kernel')['mean_test_score'].mean().sort_values()
ax.barh(range(len(param_importance)), param_importance.values)
ax.set_yticks(range(len(param_importance)))
ax.set_yticklabels(param_importance.index)
ax.set_xlabel('Mean Test Score')
ax.set_title('Kernel Performance Comparison')

plt.tight_layout()
plt.show()
```

### 2. Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define parameter distributions
param_dist = {
    'C': uniform(0.1, 100),
    'gamma': uniform(0.001, 0.1),
    'kernel': ['rbf', 'linear', 'poly']
}

# Random search
random_search = RandomizedSearchCV(
    SVC(),
    param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X, y)

print("RANDOM SEARCH RESULTS")
print("="*50)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")

# Compare Grid vs Random Search
print("\nGRID vs RANDOM SEARCH COMPARISON")
print("="*50)
print(f"Grid Search - Best Score: {grid_search.best_score_:.3f}")
print(f"Random Search - Best Score: {random_search.best_score_:.3f}")
print(f"Grid Search - Total fits: {len(grid_search.cv_results_['params'])}")
print(f"Random Search - Total fits: {len(random_search.cv_results_['params'])}")
```

### 3. Bayesian Optimization

```python
# Note: Requires installation: pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Define search space
search_space = {
    'C': Real(0.1, 100, prior='log-uniform'),
    'gamma': Real(0.001, 0.1, prior='log-uniform'),
    'kernel': Categorical(['rbf', 'linear', 'poly'])
}

# Bayesian optimization
bayes_search = BayesSearchCV(
    SVC(),
    search_space,
    n_iter=30,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X, y)

print("BAYESIAN OPTIMIZATION RESULTS")
print("="*50)
print(f"Best parameters: {bayes_search.best_params_}")
print(f"Best score: {bayes_search.best_score_:.3f}")

# Plot convergence
from skopt.plots import plot_convergence
plot_convergence(bayes_search.optimizer_results_[0])
plt.title('Bayesian Optimization Convergence')
plt.show()
```

---

## Model Persistence

### Saving and Loading Models

```python
import joblib
import pickle
from pathlib import Path

# Create a model directory
model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

# Train a model
iris = datasets.load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("MODEL PERSISTENCE")
print("="*50)

# Method 1: Using joblib (recommended for scikit-learn)
joblib_path = model_dir / 'model_joblib.pkl'
joblib.dump(model, joblib_path)
print(f"Model saved with joblib to: {joblib_path}")

# Load with joblib
loaded_model_joblib = joblib.load(joblib_path)
score_joblib = loaded_model_joblib.score(X, y)
print(f"Joblib loaded model score: {score_joblib:.3f}")

# Method 2: Using pickle
pickle_path = model_dir / 'model_pickle.pkl'
with open(pickle_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved with pickle to: {pickle_path}")

# Load with pickle
with open(pickle_path, 'rb') as f:
    loaded_model_pickle = pickle.load(f)
score_pickle = loaded_model_pickle.score(X, y)
print(f"Pickle loaded model score: {score_pickle:.3f}")

# Save pipeline with preprocessing
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X, y)

pipeline_path = model_dir / 'pipeline.pkl'
joblib.dump(pipeline, pipeline_path)
print(f"\nPipeline saved to: {pipeline_path}")

# Save model metadata
metadata = {
    'model_type': type(model).__name__,
    'n_features': X.shape[1],
    'n_classes': len(np.unique(y)),
    'feature_names': iris.feature_names,
    'target_names': list(iris.target_names),
    'training_score': model.score(X, y),
    'timestamp': pd.Timestamp.now().isoformat()
}

metadata_path = model_dir / 'model_metadata.json'
import json
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved to: {metadata_path}")
print("\nModel Metadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")
```

---

## Advanced Features

### 1. Cross-Validation Strategies

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate, 
    StratifiedKFold, RepeatedStratifiedKFold,
    cross_val_predict
)

# Load data
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_classes=2, random_state=42)

model = LogisticRegression(max_iter=1000)

print("CROSS-VALIDATION STRATEGIES")
print("="*50)

# 1. Simple cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Simple CV - Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 2. Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print(f"Stratified CV - Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 3. Repeated Stratified K-Fold
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(model, X, y, cv=rskf, scoring='accuracy')
print(f"Repeated Stratified CV - Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 4. Multiple metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
scores = cross_validate(model, X, y, cv=5, scoring=scoring)

print("\nMultiple Metrics:")
for metric in scoring:
    metric_scores = scores[f'test_{metric}']
    print(f"  {metric:10} {metric_scores.mean():.3f} (+/- {metric_scores.std():.3f})")

# 5. Cross-validation predictions
y_pred = cross_val_predict(model, X, y, cv=5)
accuracy = accuracy_score(y, y_pred)
print(f"\nCross-validated predictions accuracy: {accuracy:.3f}")
```

### 2. Feature Engineering with Scikit-learn

```python
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_simple = np.array([[1, 2], [3, 4], [5, 6]])
X_poly = poly.fit_transform(X_simple)

print("FEATURE ENGINEERING")
print("="*50)
print("Original features:")
print(X_simple)
print(f"\nPolynomial features (degree=2):")
print(X_poly)
print(f"Feature names: {poly.get_feature_names_out()}")

# Custom transformer
def log_transform(X):
    return np.log1p(np.abs(X))

log_transformer = FunctionTransformer(log_transform)
X_log = log_transformer.transform(X_simple)
print(f"\nLog-transformed features:")
print(X_log)

# Text features
texts = [
    "Machine learning is great",
    "Scikit-learn makes ML easy",
    "Python is awesome for data science"
]

# Count Vectorizer
count_vec = CountVectorizer()
X_count = count_vec.fit_transform(texts)
print(f"\nCount Vectorizer:")
print(f"Vocabulary: {count_vec.get_feature_names_out()}")
print(f"Document-term matrix shape: {X_count.shape}")

# TF-IDF
tfidf_vec = TfidfVectorizer()
X_tfidf = tfidf_vec.fit_transform(texts)
print(f"\nTF-IDF Vectorizer:")
print(f"Shape: {X_tfidf.shape}")
```

### 3. Model Inspection and Interpretation

```python
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Train a decision tree for interpretability
iris = datasets.load_iris()
X, y = iris.data, iris.target

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# Visualize decision tree
plt.figure(figsize=(15, 10))
plot_tree(dt, feature_names=iris.feature_names, 
         class_names=iris.target_names,
         filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# Permutation importance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10)

# Plot importance
fig, ax = plt.subplots(figsize=(10, 6))
sorted_idx = perm_importance.importances_mean.argsort()
ax.boxplot(perm_importance.importances[sorted_idx].T,
          vert=False, labels=np.array(iris.feature_names)[sorted_idx])
ax.set_title("Permutation Feature Importance")
ax.set_xlabel("Decrease in accuracy")
plt.show()
```

---

## Common Algorithms in Scikit-learn

### Algorithm Comparison

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Classification algorithms
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(max_iter=1000)
}

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_classes=2, random_state=42)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare algorithms
results = []
for name, clf in classifiers.items():
    # Train
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = clf.score(X_train_scaled, y_train)
    test_score = clf.score(X_test_scaled, y_test)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    
    results.append({
        'Algorithm': name,
        'Train Score': train_score,
        'Test Score': test_score,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })

# Display results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test Score', ascending=False)

print("ALGORITHM COMPARISON")
print("="*70)
print(results_df.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar plot comparison
ax = axes[0]
x = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df['Train Score'], width, label='Train', alpha=0.8)
ax.bar(x + width/2, results_df['Test Score'], width, label='Test', alpha=0.8)
ax.set_xlabel('Algorithm')
ax.set_ylabel('Accuracy')
ax.set_title('Algorithm Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Algorithm'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# Box plot of CV scores
ax = axes[1]
cv_data = []
for name, clf in classifiers.items():
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    cv_data.append(cv_scores)

ax.boxplot(cv_data, labels=[name.replace(' ', '\n') for name in classifiers.keys()])
ax.set_ylabel('Cross-Validation Score')
ax.set_title('Cross-Validation Score Distribution')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Best Practices

### 1. Reproducibility

```python
# Set random seeds for reproducibility
import random

def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    # If using TensorFlow or PyTorch, set their seeds too

set_seeds(42)

# Use random_state parameter in scikit-learn
model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

### 2. Efficient Data Handling

```python
# Use generators for large datasets
from sklearn.datasets import fetch_openml

# Load large dataset efficiently
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Use partial_fit for incremental learning
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(random_state=42)

# Train in batches
batch_size = 1000
n_batches = len(X) // batch_size

for i in range(n_batches):
    batch_X = X[i*batch_size:(i+1)*batch_size]
    batch_y = y[i*batch_size:(i+1)*batch_size]
    sgd.partial_fit(batch_X, batch_y, classes=np.unique(y))
    
    if i % 10 == 0:
        score = sgd.score(X[:5000], y[:5000])
        print(f"Batch {i}, Score: {score:.3f}")
```

### 3. Model Selection Strategy

```python
# Systematic model selection
from sklearn.model_selection import validation_curve

# Check for overfitting with validation curves
param_range = np.logspace(-3, 2, 6)
train_scores, val_scores = validation_curve(
    SVC(), X, y, param_name='C', param_range=param_range,
    cv=5, scoring='accuracy'
)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, np.mean(train_scores, axis=1), 'b-', 
         label='Training score', linewidth=2)
plt.plot(param_range, np.mean(val_scores, axis=1), 'r-', 
         label='Validation score', linewidth=2)
plt.fill_between(param_range, 
                 np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                 np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                 alpha=0.2, color='blue')
plt.fill_between(param_range,
                 np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                 np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                 alpha=0.2, color='red')
plt.xlabel('C parameter')
plt.ylabel('Score')
plt.title('Validation Curve for SVM')
plt.legend()
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Real-World Projects

### Project 1: Customer Churn Prediction

```python
# Simulate customer churn dataset
np.random.seed(42)
n_customers = 1000

# Generate features
data = pd.DataFrame({
    'tenure': np.random.randint(1, 72, n_customers),
    'monthly_charges': np.random.uniform(20, 100, n_customers),
    'total_charges': np.random.uniform(100, 7000, n_customers),
    'num_services': np.random.randint(1, 8, n_customers),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
    'payment_method': np.random.choice(['Electronic', 'Mailed check', 'Bank transfer'], n_customers),
    'paperless_billing': np.random.choice(['Yes', 'No'], n_customers)
})

# Generate target (churn) based on features
churn_probability = (
    (data['contract_type'] == 'Month-to-month').astype(int) * 0.3 +
    (data['tenure'] < 12).astype(int) * 0.2 +
    (data['monthly_charges'] > 70).astype(int) * 0.1 +
    np.random.random(n_customers) * 0.4
)
data['churn'] = (churn_probability > 0.5).astype(int)

print("CUSTOMER CHURN PREDICTION PROJECT")
print("="*50)
print(f"Dataset shape: {data.shape}")
print(f"Churn rate: {data['churn'].mean():.2%}")
print("\nFeature types:")
print(data.dtypes)

# Preprocessing pipeline
numeric_features = ['tenure', 'monthly_charges', 'total_charges', 'num_services']
categorical_features = ['contract_type', 'payment_method', 'paperless_billing']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train and evaluate
X = data.drop('churn', axis=1)
y = data['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluation
print("\nModel Performance:")
print("-"*30)
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Feature importance
feature_names = (numeric_features + 
                list(pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features)))

importances = pipeline.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(10, 6))
plt.bar(range(10), importances[indices])
plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.title('Top 10 Feature Importances - Customer Churn')
plt.tight_layout()
plt.show()
```

---

## Summary and Resources

### Key Takeaways

1. **Consistent API**: All scikit-learn estimators follow the same pattern: fit(), predict(), transform()
2. **Pipelines**: Use pipelines to avoid data leakage and create reproducible workflows
3. **Cross-validation**: Always use proper validation strategies
4. **Feature Engineering**: Crucial for model performance
5. **Hyperparameter Tuning**: Use GridSearch, RandomSearch, or Bayesian optimization
6. **Model Persistence**: Save trained models for deployment

### Scikit-learn Ecosystem

```python
# Useful scikit-learn extensions
extensions = {
    'imbalanced-learn': 'Handling imbalanced datasets',
    'scikit-optimize': 'Bayesian optimization',
    'scikit-multilearn': 'Multi-label classification',
    'scikit-survival': 'Survival analysis',
    'scikit-image': 'Image processing',
    'category_encoders': 'Advanced categorical encoding'
}

for lib, description in extensions.items():
    print(f"• {lib}: {description}")
```

### Resources

- **Official Documentation**: https://scikit-learn.org/
- **User Guide**: https://scikit-learn.org/stable/user_guide.html
- **API Reference**: https://scikit-learn.org/stable/modules/classes.html
- **Examples Gallery**: https://scikit-learn.org/stable/auto_examples/index.html
- **ML Map**: https://scikit-learn.org/stable/tutorial/machine_learning_map/

### Quick Reference Cheat Sheet

```python
# Import essentials
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

# Basic workflow
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = SomeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)

# Pipeline
pipeline = make_pipeline(StandardScaler(), SomeClassifier())
pipeline.fit(X_train, y_train)

# Grid search
param_grid = {'param1': [1, 10], 'param2': ['a', 'b']}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

---

[**Continue to Module 8: Decision Trees and Random Forests →**](08_decision_trees_random_forests.md)