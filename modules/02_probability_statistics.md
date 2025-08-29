# Module 2: Probability and Statistics 📊

[← Previous Module](01_linear_algebra.md) | [Back to Main](../README.md) | [Next Module →](03_intro_to_ml.md)

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Probability Fundamentals](#probability-fundamentals)
3. [Random Variables and Distributions](#random-variables-and-distributions)
4. [Common Probability Distributions](#common-probability-distributions)
5. [Bayes' Theorem and Bayesian Thinking](#bayes-theorem-and-bayesian-thinking)
6. [Statistical Inference](#statistical-inference)
7. [Hypothesis Testing](#hypothesis-testing)
8. [Correlation and Causation](#correlation-and-causation)
9. [Applications in Machine Learning](#applications-in-machine-learning)
10. [Practical Exercises](#practical-exercises)
11. [Summary and Key Takeaways](#summary-and-key-takeaways)

---

## Introduction

Probability and statistics form the theoretical foundation of machine learning. They help us:
- Model uncertainty in predictions
- Make inferences from data
- Evaluate model performance
- Understand the theoretical basis of algorithms

![Probability in ML](https://miro.medium.com/max/1400/1*7tKhkypJqrHPqhQQqMpWQQ.png)

### Why Probability for AI?

- **Uncertainty Quantification**: Real-world data is noisy and uncertain
- **Bayesian Methods**: Many ML algorithms are based on Bayes' theorem
- **Generative Models**: GANs, VAEs use probability distributions
- **Model Evaluation**: Statistical tests validate model performance
- **Decision Making**: Probabilistic predictions guide decisions

---

## Probability Fundamentals

### 1. Basic Concepts

**Sample Space (Ω)**: Set of all possible outcomes
**Event**: Subset of the sample space
**Probability**: Measure of likelihood (0 ≤ P(A) ≤ 1)

![Probability Basics](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Example_of_a_Venn_Diagram.svg/1200px-Example_of_a_Venn_Diagram.svg.png)

### 2. Probability Rules

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
sns.set_style("whitegrid")

# Axioms of Probability
# 1. Non-negativity: P(A) ≥ 0
# 2. Normalization: P(Ω) = 1
# 3. Additivity: P(A ∪ B) = P(A) + P(B) if A ∩ B = ∅

# Example: Dice roll
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6  # Fair die

plt.figure(figsize=(10, 4))
plt.bar(outcomes, probabilities)
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.title('Probability Distribution of a Fair Die')
plt.ylim([0, 0.3])
for i, (o, p) in enumerate(zip(outcomes, probabilities)):
    plt.text(o, p + 0.01, f'{p:.3f}', ha='center')
plt.show()
```

### 3. Conditional Probability

P(A|B) = P(A ∩ B) / P(B), where P(B) > 0

**Interpretation**: Probability of A given that B has occurred.

```python
# Example: Medical diagnosis
# Disease prevalence
P_disease = 0.01  # 1% of population has disease

# Test accuracy
P_positive_given_disease = 0.99  # Sensitivity (true positive rate)
P_negative_given_healthy = 0.95  # Specificity (true negative rate)

# Calculate P(positive test)
P_healthy = 1 - P_disease
P_positive_given_healthy = 1 - P_negative_given_healthy
P_positive = (P_positive_given_disease * P_disease + 
              P_positive_given_healthy * P_healthy)

# Bayes' theorem: P(disease|positive) 
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print(f"P(disease|positive test) = {P_disease_given_positive:.3f}")
# Only ~16.7% chance of having disease even with positive test!
```

### 4. Independence

Events A and B are independent if:
- P(A ∩ B) = P(A) × P(B)
- P(A|B) = P(A)
- P(B|A) = P(B)

```python
# Example: Coin flips are independent
num_flips = 10000
flips = np.random.choice(['H', 'T'], size=(num_flips, 2))

# Count outcomes
HH = np.sum((flips[:, 0] == 'H') & (flips[:, 1] == 'H'))
HT = np.sum((flips[:, 0] == 'H') & (flips[:, 1] == 'T'))
TH = np.sum((flips[:, 0] == 'T') & (flips[:, 1] == 'H'))
TT = np.sum((flips[:, 0] == 'T') & (flips[:, 1] == 'T'))

print(f"P(HH) = {HH/num_flips:.3f} (expected: 0.25)")
print(f"P(HT) = {HT/num_flips:.3f} (expected: 0.25)")
print(f"P(TH) = {TH/num_flips:.3f} (expected: 0.25)")
print(f"P(TT) = {TT/num_flips:.3f} (expected: 0.25)")
```

---

## Random Variables and Distributions

### 1. Random Variables

A **random variable** X is a function that maps outcomes to real numbers.

- **Discrete**: Countable values (e.g., dice roll, coin flip)
- **Continuous**: Uncountable values (e.g., height, temperature)

### 2. Probability Mass Function (PMF) - Discrete

For discrete random variable X:
P(X = x) = probability that X takes value x

```python
# Example: Binomial distribution (number of heads in n flips)
n, p = 10, 0.5  # 10 flips, fair coin
k = np.arange(0, n+1)
pmf = stats.binom.pmf(k, n, p)

plt.figure(figsize=(10, 5))
plt.bar(k, pmf)
plt.xlabel('Number of Heads')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution (n={n}, p={p})')
for i, prob in enumerate(pmf):
    plt.text(i, prob + 0.005, f'{prob:.3f}', ha='center', fontsize=8)
plt.show()
```

### 3. Probability Density Function (PDF) - Continuous

For continuous random variable X:
∫ᵃᵇ f(x)dx = P(a ≤ X ≤ b)

```python
# Example: Normal distribution
mu, sigma = 0, 1  # Mean and standard deviation
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x, mu, sigma)

plt.figure(figsize=(10, 5))
plt.plot(x, pdf, 'b-', linewidth=2, label=f'N({mu}, {sigma}²)')
plt.fill_between(x, pdf, alpha=0.3)
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Normal Distribution PDF')
plt.legend()

# Highlight 68-95-99.7 rule
for i, (lower, upper) in enumerate([(-1, 1), (-2, 2), (-3, 3)]):
    area = stats.norm.cdf(upper) - stats.norm.cdf(lower)
    plt.axvline(lower, color='r', linestyle='--', alpha=0.5)
    plt.axvline(upper, color='r', linestyle='--', alpha=0.5)
    plt.text(0, 0.1 - i*0.02, f'{area:.1%} of data', ha='center')
plt.show()
```

### 4. Cumulative Distribution Function (CDF)

F(x) = P(X ≤ x)

```python
# Compare PDF and CDF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# PDF
ax1.plot(x, pdf, 'b-', linewidth=2)
ax1.fill_between(x, pdf, alpha=0.3)
ax1.set_title('Probability Density Function')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')

# CDF
cdf = stats.norm.cdf(x, mu, sigma)
ax2.plot(x, cdf, 'r-', linewidth=2)
ax2.set_title('Cumulative Distribution Function')
ax2.set_xlabel('x')
ax2.set_ylabel('F(x) = P(X ≤ x)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5. Expected Value and Variance

**Expected Value (Mean)**: E[X] = Σ x·P(X=x) or ∫ x·f(x)dx
**Variance**: Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
**Standard Deviation**: σ = √Var(X)

```python
# Example: Calculate statistics for different distributions
distributions = {
    'Uniform(0,1)': stats.uniform(0, 1),
    'Normal(0,1)': stats.norm(0, 1),
    'Exponential(λ=1)': stats.expon(scale=1),
    'Beta(2,5)': stats.beta(2, 5)
}

for name, dist in distributions.items():
    mean = dist.mean()
    var = dist.var()
    std = dist.std()
    print(f"{name:20} Mean={mean:.3f}, Var={var:.3f}, Std={std:.3f}")
```

---

## Common Probability Distributions

### 1. Discrete Distributions

#### Bernoulli Distribution
Single trial with binary outcome (success/failure)

```python
# Bernoulli: Coin flip
p = 0.7  # Probability of success
bernoulli = stats.bernoulli(p)

# Simulate 1000 trials
samples = bernoulli.rvs(1000)
print(f"Empirical mean: {np.mean(samples):.3f} (Expected: {p:.3f})")
print(f"Empirical variance: {np.var(samples):.3f} (Expected: {p*(1-p):.3f})")
```

#### Binomial Distribution
Number of successes in n independent Bernoulli trials

![Binomial Distribution](https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Binomial_distribution_pmf.svg/1200px-Binomial_distribution_pmf.svg.png)

```python
# Binomial: Number of successful A/B test conversions
n_visitors = 100
conversion_rate = 0.1
k = np.arange(0, 30)

pmf = stats.binom.pmf(k, n_visitors, conversion_rate)

plt.figure(figsize=(10, 5))
plt.bar(k, pmf)
plt.xlabel('Number of Conversions')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution: P(conversions out of {n_visitors} visitors)')
plt.axvline(n_visitors * conversion_rate, color='r', linestyle='--', 
            label=f'Expected: {n_visitors * conversion_rate}')
plt.legend()
plt.show()
```

#### Poisson Distribution
Number of events in fixed interval (rare events)

```python
# Poisson: Website visits per hour
lambda_rate = 5  # Average visits per hour
k = np.arange(0, 15)
pmf = stats.poisson.pmf(k, lambda_rate)

plt.figure(figsize=(10, 5))
plt.bar(k, pmf)
plt.xlabel('Number of Visits')
plt.ylabel('Probability')
plt.title(f'Poisson Distribution (λ={lambda_rate})')
for i in range(len(k)):
    if pmf[i] > 0.01:
        plt.text(k[i], pmf[i] + 0.005, f'{pmf[i]:.3f}', ha='center', fontsize=8)
plt.show()
```

### 2. Continuous Distributions

#### Normal (Gaussian) Distribution
The most important distribution in statistics!

```python
# Visualize different normal distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Different means
ax = axes[0, 0]
for mu in [-2, 0, 2]:
    x = np.linspace(-6, 6, 1000)
    y = stats.norm.pdf(x, mu, 1)
    ax.plot(x, y, label=f'μ={mu}, σ=1')
ax.set_title('Effect of Mean (μ)')
ax.legend()

# Different standard deviations
ax = axes[0, 1]
for sigma in [0.5, 1, 2]:
    x = np.linspace(-6, 6, 1000)
    y = stats.norm.pdf(x, 0, sigma)
    ax.plot(x, y, label=f'μ=0, σ={sigma}')
ax.set_title('Effect of Standard Deviation (σ)')
ax.legend()

# Standard normal and Z-scores
ax = axes[1, 0]
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)
ax.plot(x, y, 'b-', linewidth=2)
ax.fill_between(x, y, alpha=0.3)
ax.set_title('Standard Normal Distribution')
ax.set_xlabel('Z-score')

# Q-Q plot to check normality
ax = axes[1, 1]
samples = np.random.normal(0, 1, 1000)
stats.probplot(samples, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Check for Normality)')

plt.tight_layout()
plt.show()
```

#### Exponential Distribution
Time between events in Poisson process

```python
# Exponential: Time between customer arrivals
lambda_rate = 2  # Average 2 arrivals per unit time
x = np.linspace(0, 5, 1000)
pdf = stats.expon.pdf(x, scale=1/lambda_rate)

plt.figure(figsize=(10, 5))
plt.plot(x, pdf, 'g-', linewidth=2, label=f'λ={lambda_rate}')
plt.fill_between(x, pdf, alpha=0.3)
plt.xlabel('Time Between Arrivals')
plt.ylabel('Probability Density')
plt.title('Exponential Distribution')
plt.legend()

# Memoryless property
print(f"P(X > 1) = {1 - stats.expon.cdf(1, scale=1/lambda_rate):.3f}")
print(f"P(X > 2 | X > 1) = {1 - stats.expon.cdf(1, scale=1/lambda_rate):.3f}")
print("Same probability! (Memoryless property)")
plt.show()
```

#### Beta Distribution
Probability of probabilities (used in Bayesian inference)

```python
# Beta distribution: Modeling uncertainty in conversion rates
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
x = np.linspace(0, 1, 1000)

# Different shape parameters
params = [(0.5, 0.5), (2, 2), (2, 5), (5, 2)]
titles = ['U-shaped', 'Symmetric', 'Right-skewed', 'Left-skewed']

for ax, (a, b), title in zip(axes.flat, params, titles):
    y = stats.beta.pdf(x, a, b)
    ax.plot(x, y, linewidth=2)
    ax.fill_between(x, y, alpha=0.3)
    ax.set_title(f'{title}: Beta({a}, {b})')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Density')
    ax.set_ylim([0, max(y) * 1.1])

plt.tight_layout()
plt.show()
```

---

## Bayes' Theorem and Bayesian Thinking

### Bayes' Theorem

P(A|B) = P(B|A) × P(A) / P(B)

Where:
- P(A|B): Posterior probability
- P(B|A): Likelihood
- P(A): Prior probability
- P(B): Evidence

![Bayes Theorem](https://miro.medium.com/max/1400/1*LB-G6WBuswEfpg20FMighA.png)

### Bayesian Updating

```python
# Bayesian coin flip inference
def bayesian_coin_flip(flips, prior_a=1, prior_b=1):
    """
    Update belief about coin fairness using Beta-Binomial conjugacy
    """
    heads = sum(1 for f in flips if f == 'H')
    tails = len(flips) - heads
    
    # Update Beta distribution parameters
    posterior_a = prior_a + heads
    posterior_b = prior_b + tails
    
    return posterior_a, posterior_b

# Simulate coin flips
true_p = 0.7  # Biased coin
flips = np.random.choice(['H', 'T'], size=100, p=[true_p, 1-true_p])

# Plot belief evolution
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
x = np.linspace(0, 1, 1000)

n_flips = [0, 10, 20, 50, 100]
for i, n in enumerate(n_flips):
    ax = axes[i//3, i%3]
    
    # Get posterior
    a, b = bayesian_coin_flip(flips[:n])
    
    # Plot prior (gray) and posterior (blue)
    if n == 0:
        ax.plot(x, stats.beta.pdf(x, 1, 1), 'gray', label='Prior')
    else:
        ax.plot(x, stats.beta.pdf(x, 1, 1), 'gray', alpha=0.3)
    
    posterior = stats.beta.pdf(x, a, b)
    ax.plot(x, posterior, 'b-', linewidth=2, label=f'After {n} flips')
    ax.fill_between(x, posterior, alpha=0.3)
    
    # Mark true value
    ax.axvline(true_p, color='r', linestyle='--', label=f'True p={true_p}')
    
    # Mark posterior mean
    post_mean = a / (a + b)
    ax.axvline(post_mean, color='g', linestyle='--', 
               label=f'Posterior mean={post_mean:.2f}')
    
    ax.set_title(f'After {n} flips')
    ax.set_xlabel('p (probability of heads)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

# Remove empty subplot
fig.delaxes(axes[1, 2])
plt.tight_layout()
plt.show()
```

### Naive Bayes Classifier

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predictions with probabilities
y_pred = nb.predict(X_test)
y_proba = nb.predict_proba(X_test)

# Visualize predictions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
im = ax1.imshow(cm, cmap='Blues')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix')
for i in range(3):
    for j in range(3):
        ax1.text(j, i, cm[i, j], ha='center', va='center')

# Probability distribution for first 10 test samples
ax2.imshow(y_proba[:10], aspect='auto', cmap='YlOrRd')
ax2.set_xlabel('Class')
ax2.set_ylabel('Sample')
ax2.set_title('Class Probabilities (First 10 Test Samples)')
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(iris.target_names)

plt.tight_layout()
plt.show()

print(f"Accuracy: {nb.score(X_test, y_test):.3f}")
```

---

## Statistical Inference

### 1. Point Estimation

Estimating population parameters from sample data.

```python
# Maximum Likelihood Estimation (MLE) for Normal distribution
def mle_normal(data):
    """
    MLE for normal distribution parameters
    """
    n = len(data)
    mu_mle = np.mean(data)
    sigma_mle = np.sqrt(np.sum((data - mu_mle)**2) / n)
    return mu_mle, sigma_mle

# Generate sample from known distribution
true_mu, true_sigma = 5, 2
sample = np.random.normal(true_mu, true_sigma, 100)

# Estimate parameters
mu_est, sigma_est = mle_normal(sample)

print(f"True parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates: μ={mu_est:.3f}, σ={sigma_est:.3f}")

# Visualize
x = np.linspace(0, 10, 1000)
plt.figure(figsize=(10, 5))
plt.hist(sample, bins=20, density=True, alpha=0.5, label='Sample data')
plt.plot(x, stats.norm.pdf(x, true_mu, true_sigma), 'r-', 
         label=f'True: N({true_mu}, {true_sigma}²)', linewidth=2)
plt.plot(x, stats.norm.pdf(x, mu_est, sigma_est), 'b--', 
         label=f'MLE: N({mu_est:.2f}, {sigma_est:.2f}²)', linewidth=2)
plt.legend()
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Maximum Likelihood Estimation')
plt.show()
```

### 2. Confidence Intervals

Range of plausible values for population parameter.

```python
def confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for mean
    """
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error
    
    # t-distribution for small samples
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean - margin, mean + margin

# Bootstrap confidence interval
def bootstrap_ci(data, statistic=np.mean, n_bootstrap=1000, confidence=0.95):
    """
    Bootstrap confidence interval
    """
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(resample))
    
    # Percentile method
    lower = np.percentile(bootstrap_stats, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 + confidence) / 2 * 100)
    
    return lower, upper, bootstrap_stats

# Example
sample_data = np.random.normal(100, 15, 50)

# Parametric CI
ci_lower, ci_upper = confidence_interval(sample_data)
print(f"95% Confidence Interval (parametric): [{ci_lower:.2f}, {ci_upper:.2f}]")

# Bootstrap CI
boot_lower, boot_upper, boot_dist = bootstrap_ci(sample_data)
print(f"95% Confidence Interval (bootstrap): [{boot_lower:.2f}, {boot_upper:.2f}]")

# Visualize
plt.figure(figsize=(10, 5))
plt.hist(boot_dist, bins=30, density=True, alpha=0.6, label='Bootstrap distribution')
plt.axvline(np.mean(sample_data), color='r', linestyle='-', linewidth=2, label='Sample mean')
plt.axvline(boot_lower, color='g', linestyle='--', label='95% CI')
plt.axvline(boot_upper, color='g', linestyle='--')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of Sample Mean')
plt.legend()
plt.show()
```

---

## Hypothesis Testing

### 1. Framework

1. **Null Hypothesis (H₀)**: Default assumption
2. **Alternative Hypothesis (H₁)**: What we want to prove
3. **Test Statistic**: Measure of evidence
4. **P-value**: Probability of observing data if H₀ is true
5. **Decision**: Reject H₀ if p-value < α (significance level)

### 2. Types of Errors

- **Type I Error (α)**: Rejecting true H₀ (False Positive)
- **Type II Error (β)**: Not rejecting false H₀ (False Negative)
- **Power (1-β)**: Probability of detecting true effect

```python
# Simulation of hypothesis testing
def simulate_hypothesis_test(n_simulations=10000, effect_size=0):
    """
    Simulate t-tests under null and alternative hypotheses
    """
    p_values = []
    
    for _ in range(n_simulations):
        # Generate data
        control = np.random.normal(0, 1, 30)
        treatment = np.random.normal(effect_size, 1, 30)
        
        # Perform t-test
        _, p_value = stats.ttest_ind(control, treatment)
        p_values.append(p_value)
    
    return np.array(p_values)

# Simulate under null (no effect)
p_values_null = simulate_hypothesis_test(effect_size=0)

# Simulate under alternative (with effect)
p_values_alt = simulate_hypothesis_test(effect_size=0.5)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# P-value distribution under null
ax1.hist(p_values_null, bins=50, density=True, alpha=0.7, label='H₀ true')
ax1.axvline(0.05, color='r', linestyle='--', label='α=0.05')
ax1.set_xlabel('P-value')
ax1.set_ylabel('Density')
ax1.set_title('P-value Distribution Under Null')
ax1.legend()

# P-value distribution under alternative
ax2.hist(p_values_alt, bins=50, density=True, alpha=0.7, 
         color='orange', label='H₁ true')
ax2.axvline(0.05, color='r', linestyle='--', label='α=0.05')
ax2.set_xlabel('P-value')
ax2.set_ylabel('Density')
ax2.set_title('P-value Distribution Under Alternative')
ax2.legend()

plt.tight_layout()
plt.show()

# Calculate error rates
alpha = 0.05
type_i_error = np.mean(p_values_null < alpha)
power = np.mean(p_values_alt < alpha)

print(f"Type I Error Rate: {type_i_error:.3f} (Expected: {alpha})")
print(f"Statistical Power: {power:.3f}")
```

### 3. Common Statistical Tests

```python
# t-test: Compare means
group1 = np.random.normal(100, 10, 50)
group2 = np.random.normal(105, 10, 50)

t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")

# Chi-square test: Test independence
observed = np.array([[10, 20, 30], [15, 25, 20]])
chi2, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"Chi-square test: χ²={chi2:.3f}, p={p_value:.3f}")

# ANOVA: Compare multiple groups
group1 = np.random.normal(100, 10, 30)
group2 = np.random.normal(102, 10, 30)
group3 = np.random.normal(105, 10, 30)

f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"ANOVA: F={f_stat:.3f}, p={p_value:.3f}")

# Kolmogorov-Smirnov test: Test distribution
data = np.random.normal(0, 1, 100)
ks_stat, p_value = stats.kstest(data, 'norm')
print(f"KS test for normality: KS={ks_stat:.3f}, p={p_value:.3f}")
```

### 4. Multiple Testing Correction

```python
# Multiple testing problem
n_tests = 20
p_values = []

for i in range(n_tests):
    # All from null hypothesis (no real effect)
    group1 = np.random.normal(0, 1, 30)
    group2 = np.random.normal(0, 1, 30)
    _, p = stats.ttest_ind(group1, group2)
    p_values.append(p)

p_values = np.array(p_values)

# Bonferroni correction
alpha = 0.05
bonferroni_alpha = alpha / n_tests

# Benjamini-Hochberg (FDR control)
from statsmodels.stats.multitest import multipletests
reject_bh, p_adjusted_bh, _, _ = multipletests(p_values, method='fdr_bh')

print(f"Uncorrected: {np.sum(p_values < alpha)} significant")
print(f"Bonferroni: {np.sum(p_values < bonferroni_alpha)} significant")
print(f"Benjamini-Hochberg: {np.sum(reject_bh)} significant")
```

---

## Correlation and Causation

### 1. Correlation Measures

```python
# Generate correlated data
np.random.seed(42)
n = 100

# Different types of relationships
x = np.random.randn(n)
y_linear = 2 * x + np.random.randn(n) * 0.5
y_quadratic = x**2 + np.random.randn(n) * 0.5
y_random = np.random.randn(n)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Linear correlation
ax = axes[0, 0]
ax.scatter(x, y_linear, alpha=0.6)
r_pearson = np.corrcoef(x, y_linear)[0, 1]
ax.set_title(f'Linear: r={r_pearson:.3f}')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Quadratic (non-linear)
ax = axes[0, 1]
ax.scatter(x, y_quadratic, alpha=0.6)
r_pearson = np.corrcoef(x, y_quadratic)[0, 1]
r_spearman = stats.spearmanr(x, y_quadratic)[0]
ax.set_title(f'Quadratic: Pearson={r_pearson:.3f}, Spearman={r_spearman:.3f}')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# No correlation
ax = axes[0, 2]
ax.scatter(x, y_random, alpha=0.6)
r_pearson = np.corrcoef(x, y_random)[0, 1]
ax.set_title(f'Random: r={r_pearson:.3f}')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Correlation matrix heatmap
ax = axes[1, 0]
data = np.column_stack([x, y_linear, y_quadratic, y_random])
corr_matrix = np.corrcoef(data.T)
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])
ax.set_xticklabels(['X', 'Y_linear', 'Y_quad', 'Y_random'])
ax.set_yticklabels(['X', 'Y_linear', 'Y_quad', 'Y_random'])
ax.set_title('Correlation Matrix')
plt.colorbar(im, ax=ax)

# Anscombe's quartet - same statistics, different relationships
ax = axes[1, 1]
anscombe = sns.load_dataset("anscombe")
for dataset in ['I', 'II', 'III', 'IV']:
    data = anscombe[anscombe['dataset'] == dataset]
    ax.scatter(data['x'], data['y'], label=dataset, alpha=0.7)
ax.set_title("Anscombe's Quartet (all r≈0.816)")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# Spurious correlation
ax = axes[1, 2]
years = np.arange(2000, 2020)
ice_cream_sales = 100 + 5 * years + np.random.randn(20) * 10
shark_attacks = 50 + 2.5 * years + np.random.randn(20) * 5
ax.plot(years, ice_cream_sales / 10, 'b-', label='Ice Cream Sales')
ax.plot(years, shark_attacks / 5, 'r-', label='Shark Attacks')
ax.set_title('Spurious Correlation')
ax.set_xlabel('Year')
ax.set_ylabel('Normalized Value')
ax.legend()

plt.tight_layout()
plt.show()
```

### 2. Simpson's Paradox

```python
# Simpson's Paradox example
# Overall correlation can reverse when controlling for groups

# Generate data
np.random.seed(42)
n_per_group = 50

# Group 1: Low X, Low Y
x1 = np.random.uniform(0, 3, n_per_group)
y1 = 0.5 * x1 + np.random.normal(0, 0.2, n_per_group)

# Group 2: High X, High Y
x2 = np.random.uniform(3, 6, n_per_group)
y2 = 0.5 * x2 + np.random.normal(2, 0.2, n_per_group)

# Combine
x_all = np.concatenate([x1, x2])
y_all = np.concatenate([y1, y2])
groups = np.concatenate([np.zeros(n_per_group), np.ones(n_per_group)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Overall trend
ax1.scatter(x_all, y_all, alpha=0.5, c='gray')
z = np.polyfit(x_all, y_all, 1)
p = np.poly1d(z)
ax1.plot(x_all, p(x_all), 'k-', linewidth=2, label='Overall trend')
ax1.set_title("Simpson's Paradox: Aggregate View")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()

# By group
colors = ['blue', 'red']
for i, (xi, yi, color) in enumerate([(x1, y1, 'blue'), (x2, y2, 'red')]):
    ax2.scatter(xi, yi, alpha=0.5, c=color, label=f'Group {i+1}')
    z = np.polyfit(xi, yi, 1)
    p = np.poly1d(z)
    x_line = np.linspace(xi.min(), xi.max(), 100)
    ax2.plot(x_line, p(x_line), color=color, linewidth=2)

ax2.set_title("Simpson's Paradox: Grouped View")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend()

plt.tight_layout()
plt.show()

print("Correlation coefficients:")
print(f"Overall: r={np.corrcoef(x_all, y_all)[0,1]:.3f}")
print(f"Group 1: r={np.corrcoef(x1, y1)[0,1]:.3f}")
print(f"Group 2: r={np.corrcoef(x2, y2)[0,1]:.3f}")
```

---

## Applications in Machine Learning

### 1. Probabilistic Classification

```python
# Logistic regression as probabilistic classifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)

# Train logistic regression
lr = LogisticRegression()
lr.fit(X, y)

# Create mesh for decision boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict probabilities
Z = lr.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 5))
plt.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.8)
plt.colorbar(label='P(Class 1)')
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', edgecolor='k', label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', edgecolor='k', label='Class 1')
plt.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Probabilistic Classification with Logistic Regression')
plt.legend()
plt.show()
```

### 2. Gaussian Mixture Models

```python
from sklearn.mixture import GaussianMixture

# Generate synthetic data
np.random.seed(42)
n_samples = 300
component1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//3)
component2 = np.random.multivariate_normal([3, 3], [[1, -0.5], [-0.5, 1]], n_samples//3)
component3 = np.random.multivariate_normal([0, 4], [[0.5, 0], [0, 0.5]], n_samples//3)
X = np.vstack([component1, component2, component3])

# Fit GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# True components
ax1.scatter(component1[:, 0], component1[:, 1], alpha=0.5, label='Component 1')
ax1.scatter(component2[:, 0], component2[:, 1], alpha=0.5, label='Component 2')
ax1.scatter(component3[:, 0], component3[:, 1], alpha=0.5, label='Component 3')
ax1.set_title('True Components')
ax1.legend()

# GMM predictions
colors = ['red', 'green', 'blue']
for i in range(3):
    mask = labels == i
    ax2.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.5, label=f'Cluster {i}')

# Plot Gaussian ellipses
from matplotlib.patches import Ellipse
for i in range(3):
    mean = gmm.means_[i]
    cov = gmm.covariances_[i]
    v, w = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(w[1, 0], w[0, 0]))
    
    for std in [1, 2]:
        ell = Ellipse(mean, 2*std*np.sqrt(v[0]), 2*std*np.sqrt(v[1]),
                     angle=angle, facecolor='none', 
                     edgecolor=colors[i], linewidth=2, alpha=0.5)
        ax2.add_patch(ell)

ax2.set_title('GMM Clustering')
ax2.legend()

plt.tight_layout()
plt.show()
```

### 3. A/B Testing Framework

```python
def ab_test_analysis(control, treatment, alpha=0.05):
    """
    Comprehensive A/B test analysis
    """
    n_control = len(control)
    n_treatment = len(treatment)
    
    # Basic statistics
    mean_control = np.mean(control)
    mean_treatment = np.mean(treatment)
    std_control = np.std(control, ddof=1)
    std_treatment = np.std(treatment, ddof=1)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n_control-1)*std_control**2 + 
                          (n_treatment-1)*std_treatment**2) / 
                         (n_control + n_treatment - 2))
    cohens_d = (mean_treatment - mean_control) / pooled_std
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(control, treatment)
    
    # Confidence interval for difference
    se_diff = pooled_std * np.sqrt(1/n_control + 1/n_treatment)
    ci_margin = se_diff * stats.t.ppf(1 - alpha/2, n_control + n_treatment - 2)
    diff = mean_treatment - mean_control
    ci_lower = diff - ci_margin
    ci_upper = diff + ci_margin
    
    # Power analysis
    from statsmodels.stats.power import ttest_power
    power = ttest_power(cohens_d, n_control, alpha, alternative='two-sided')
    
    results = {
        'control_mean': mean_control,
        'treatment_mean': mean_treatment,
        'difference': diff,
        'relative_change': diff / mean_control * 100,
        'p_value': p_value,
        'significant': p_value < alpha,
        'cohens_d': cohens_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'power': power
    }
    
    return results

# Simulate A/B test
np.random.seed(42)
control = np.random.normal(100, 20, 1000)  # Control group
treatment = np.random.normal(103, 20, 1000)  # Treatment with small effect

results = ab_test_analysis(control, treatment)

print("A/B Test Results:")
print("-" * 40)
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key:20}: {value:.4f}")
    else:
        print(f"{key:20}: {value}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Distributions
ax = axes[0]
ax.hist(control, bins=30, alpha=0.5, label='Control', density=True)
ax.hist(treatment, bins=30, alpha=0.5, label='Treatment', density=True)
ax.axvline(np.mean(control), color='blue', linestyle='--')
ax.axvline(np.mean(treatment), color='orange', linestyle='--')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Distribution Comparison')
ax.legend()

# Effect size with CI
ax = axes[1]
ax.errorbar([0], [results['difference']], 
           yerr=[[results['difference']-results['ci_lower']], 
                 [results['ci_upper']-results['difference']]], 
           fmt='o', markersize=10, capsize=10, capthick=2)
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlim([-0.5, 0.5])
ax.set_xticks([0])
ax.set_xticklabels(['Treatment - Control'])
ax.set_ylabel('Difference')
ax.set_title(f"Effect Size (p={results['p_value']:.4f})")

# Sequential testing simulation
ax = axes[2]
p_values = []
sample_sizes = range(50, 1001, 50)
for n in sample_sizes:
    _, p = stats.ttest_ind(control[:n], treatment[:n])
    p_values.append(p)

ax.plot(sample_sizes, p_values, 'b-')
ax.axhline(0.05, color='r', linestyle='--', label='α=0.05')
ax.set_xlabel('Sample Size')
ax.set_ylabel('P-value')
ax.set_title('P-value vs Sample Size')
ax.legend()

plt.tight_layout()
plt.show()
```

---

## Practical Exercises

### Exercise 1: Probability Calculations

```python
# Problem: Birthday Paradox
def birthday_paradox(n_people, n_simulations=10000):
    """
    Estimate probability that at least 2 people share a birthday
    """
    matches = 0
    for _ in range(n_simulations):
        birthdays = np.random.randint(1, 366, n_people)
        if len(np.unique(birthdays)) < n_people:
            matches += 1
    return matches / n_simulations

# Calculate for different group sizes
group_sizes = range(2, 60)
probabilities = [birthday_paradox(n) for n in group_sizes]

plt.figure(figsize=(10, 5))
plt.plot(group_sizes, probabilities, 'b-', linewidth=2)
plt.axhline(0.5, color='r', linestyle='--', label='50% probability')
plt.axvline(23, color='g', linestyle='--', label='23 people')
plt.xlabel('Number of People')
plt.ylabel('Probability of Shared Birthday')
plt.title('Birthday Paradox')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Exercise 2: Central Limit Theorem

```python
# Demonstrate CLT with different distributions
def demonstrate_clt(distribution, n_samples=1000, sample_sizes=[1, 5, 30, 100]):
    """
    Show how sample means approach normal distribution
    """
    fig, axes = plt.subplots(2, len(sample_sizes), figsize=(15, 8))
    
    for i, n in enumerate(sample_sizes):
        # Generate sample means
        sample_means = []
        for _ in range(n_samples):
            sample = distribution.rvs(n)
            sample_means.append(np.mean(sample))
        
        # Plot histogram
        ax = axes[0, i]
        ax.hist(sample_means, bins=30, density=True, alpha=0.7)
        
        # Overlay normal distribution
        mean = np.mean(sample_means)
        std = np.std(sample_means)
        x = np.linspace(mean - 4*std, mean + 4*std, 100)
        ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2)
        ax.set_title(f'n={n}')
        ax.set_xlabel('Sample Mean')
        if i == 0:
            ax.set_ylabel('Density')
        
        # Q-Q plot
        ax = axes[1, i]
        stats.probplot(sample_means, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot (n={n})')
    
    plt.suptitle(f'Central Limit Theorem: {distribution.dist.name} Distribution', 
                 fontsize=14)
    plt.tight_layout()
    plt.show()

# Test with different distributions
distributions = [
    stats.expon(scale=1),  # Exponential (skewed)
    stats.uniform(0, 1),    # Uniform
    stats.beta(2, 5)        # Beta (asymmetric)
]

for dist in distributions:
    demonstrate_clt(dist)
```

### Exercise 3: Bayesian A/B Testing

```python
def bayesian_ab_test(successes_a, trials_a, successes_b, trials_b, 
                     prior_alpha=1, prior_beta=1):
    """
    Bayesian A/B test using Beta-Binomial model
    """
    # Posterior parameters
    alpha_a = prior_alpha + successes_a
    beta_a = prior_beta + trials_a - successes_a
    alpha_b = prior_alpha + successes_b
    beta_b = prior_beta + trials_b - successes_b
    
    # Sample from posteriors
    samples_a = np.random.beta(alpha_a, beta_a, 100000)
    samples_b = np.random.beta(alpha_b, beta_b, 100000)
    
    # Probability B > A
    prob_b_better = np.mean(samples_b > samples_a)
    
    # Expected loss
    loss_choosing_a = np.maximum(samples_b - samples_a, 0).mean()
    loss_choosing_b = np.maximum(samples_a - samples_b, 0).mean()
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Posterior distributions
    ax = axes[0]
    x = np.linspace(0, 1, 1000)
    ax.plot(x, stats.beta.pdf(x, alpha_a, beta_a), 'b-', label='Variant A')
    ax.plot(x, stats.beta.pdf(x, alpha_b, beta_b), 'r-', label='Variant B')
    ax.set_xlabel('Conversion Rate')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distributions')
    ax.legend()
    
    # Difference distribution
    ax = axes[1]
    diff = samples_b - samples_a
    ax.hist(diff, bins=50, density=True, alpha=0.7)
    ax.axvline(0, color='r', linestyle='--')
    ax.set_xlabel('Difference (B - A)')
    ax.set_ylabel('Density')
    ax.set_title(f'P(B > A) = {prob_b_better:.3f}')
    
    # Expected loss
    ax = axes[2]
    ax.bar(['Choose A', 'Choose B'], [loss_choosing_a, loss_choosing_b])
    ax.set_ylabel('Expected Loss')
    ax.set_title('Expected Loss by Decision')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'prob_b_better': prob_b_better,
        'loss_choosing_a': loss_choosing_a,
        'loss_choosing_b': loss_choosing_b,
        'recommended': 'B' if loss_choosing_b < loss_choosing_a else 'A'
    }

# Example: Email campaign A/B test
results = bayesian_ab_test(
    successes_a=120, trials_a=1000,  # 12% conversion
    successes_b=145, trials_b=1000   # 14.5% conversion
)

print("Bayesian A/B Test Results:")
for key, value in results.items():
    print(f"{key}: {value}")
```

---

## Summary and Key Takeaways

### 🎯 Core Concepts Mastered

1. **Probability Fundamentals**
   - Sample spaces, events, and probability rules
   - Conditional probability and independence
   - Bayes' theorem and Bayesian thinking

2. **Distributions**
   - Discrete: Bernoulli, Binomial, Poisson
   - Continuous: Normal, Exponential, Beta
   - Understanding PMF, PDF, CDF

3. **Statistical Inference**
   - Point estimation and confidence intervals
   - Maximum likelihood estimation
   - Bootstrap methods

4. **Hypothesis Testing**
   - Framework and types of errors
   - Common statistical tests
   - Multiple testing corrections

5. **Practical Applications**
   - A/B testing methodology
   - Bayesian vs Frequentist approaches
   - Understanding correlation vs causation

### 💡 Key Insights for ML

1. **Uncertainty is Fundamental**: ML models don't just make predictions; they quantify uncertainty.

2. **Distributions Everywhere**: From initialization (normal) to regularization (Laplace) to outputs (softmax).

3. **Bayesian Thinking**: Prior knowledge + data = posterior beliefs. This is how many ML algorithms work.

4. **Statistical Validation**: Proper testing ensures models generalize beyond training data.

5. **Causation ≠ Correlation**: ML finds patterns, but understanding causation requires careful experimental design.


[**Continue to Module 3: Introduction to Machine Learning →**](03_intro_to_ml.md)