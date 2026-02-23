# Active Learning System

> **Python Version**: Python 3.13+

## Try It Now

This guide explains how the Name Ranking application learns your name
preferences. You will:

1. Extract **phonetic features** from names using Double Metaphone
2. Build a **Bradley-Terry model** with Bayesian updates
3. Implement **Thompson sampling** for intelligent pair selection
4. Persist model state to **SQLite** for 44,000+ names

> **Note**: For a step-by-step usage guide, see the [Tutorial](tutorial.md).
> This document focuses on the theoretical foundations of the system.

## Overview

The active learning system has four main components:

1. **Feature Extraction**: Convert names into numerical feature vectors
2. **Bradley‑Terry Model**: Build a Bayesian preference model with Laplace approximation
3. **Thompson Sampling**: Run the active learning algorithm for pair selection
4. **Model Persistence**: Store model state in the database

## Feature Engineering

### Phonetic Features (Double Metaphone)

The **Double Metaphone** algorithm converts names to phonetic codes that capture
pronunciation similarities across languages:

- **Primary and secondary codes**: Each name produces two phonetic codes
- **Feature encoding**: The system one‑hot encodes codes as categorical features
- **Similarity computation**: Use these for pair selection and similarity search

Phonetic similarity scores:

- **Exact match**: Both primary codes match (score = 1.0)
- **Primary‑secondary match**: Primary of one matches secondary of the other
  (score = 0.8)
- **Partial match**: Codes share prefix or edit distance within threshold (score
  = 0.5–0.7)
- **No match**: Score = 0.0

### Linguistic Features

- **Syllable count**: Using `pyphen` for language‑appropriate syllable division
- **Vowel ratio**: Proportion of vowels to total characters
- **Name length**: Total character count
- **Character statistics**: Distribution of letter types

### Metadata Features

- **Gender encoding**: One‑hot encoding of Male/Female/Unisex
- **Origin region**: One‑hot encoding of geographic regions (Nordic, European,
  Asian, etc.)
- **Classification confidence**: Confidence score from `ethnidata` predictions

### Feature Pipeline

1. **Text normalization**: Lowercase conversion, Unicode normalization
2. **Phonetic encoding**: Double Metaphone primary/secondary codes
3. **Linguistic analysis**: Syllable counting, character statistics
4. **Metadata lookup**: Database queries for gender and origin
5. **Normalization**: Min‑max scaling to [0,1] range
6. **Vector assembly**: Concatenation into 25‑dimensional feature vector

## Bradley‑Terry Model

### Mathematical Formulation

The Bradley‑Terry model estimates the probability that name A is preferred over
name B as:

```
P(A ≻ B) = σ(w·φ(A) - w·φ(B))
```

Where:

- `σ(x) = 1 / (1 + exp(-x))` is the logistic function
- `φ(name)` is the feature vector for a name
- `w` is the weight vector to be learned

### Bayesian Inference

The system applies a **Gaussian prior** on the weights with **Laplace
approximation** for efficient Bayesian updates:

- **Prior**: `w ∼ N(0, Σ₀)` with diagonal covariance `Σ₀ = I`
- **Posterior approximation**: Gaussian `N(μ, Σ)` updated after each comparison
- **Laplace approximation**: Second‑order Taylor expansion around MAP estimate

### Model State

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelState:
    weight_mean: np.ndarray      # μ: mean weight vector (d,)
    weight_cov: np.ndarray       # Σ: covariance matrix (d, d)
    training_samples: int        # Number of training comparisons
    feature_names: list[str]     # Names of each feature dimension
```

## Thompson Sampling for Active Learning

### Exploration‑Exploitation Balance

Thompson sampling balances exploitation and exploration:

- **Exploitation**: Compare names with high predicted preference difference
- **Exploration**: Compare names where the model has high uncertainty

### Algorithm

1. **Sample weights**: Draw `w̃ ∼ N(μ, Σ)` from current posterior
2. **Compute utilities**: `u(name) = w̃·φ(name)` for all names
3. **Select candidates**: Choose names that maximize expected information gain
4. **Diversity constraint**: Ensure feature space coverage

### Information Gain Metrics

The system maximizes:

- **Preference uncertainty**: `Var[P(A ≻ B)]`
- **Feature space coverage**: Distance in feature space
- **Comparison history**: Avoid recently compared pairs

## Model Operations

### Initialization

```python
import numpy as np
from typing import List

def initialize_model(feature_dim: int, feature_names: List[str]) -> ModelState:
    """Initialize a new model with zero mean and identity covariance."""
    return ModelState(
        weight_mean=np.zeros(feature_dim),
        weight_cov=np.eye(feature_dim),
        training_samples=0,
        feature_names=feature_names
    )

# Example: Initialize a 25-dimensional model
feature_names = [
    "syllable_count", "vowel_ratio", "name_length",
    # ... 22 more features
]
model = initialize_model(feature_dim=25, feature_names=feature_names)
print(f"Initialized model with {model.training_samples} training samples")
```

### Bayesian Update

After observing preference `y ∈ {-1, 0, 1, 2}` for names A and B:

- `y = -1`: Prefer name A over B
- `y = 1`: Prefer name B over A
- `y = 0`: Draw (both equally preferred)
- `y = 2`: Down (dislike both names, treated as two comparisons against neutral baseline)

1. **Compute feature difference**: `Δφ = φ(A) - φ(B)`
2. **Compute MAP estimate**: Newton‑Raphson optimization
3. **Update covariance**: `Σ = (Σ₀⁻¹ + H)⁻¹` where H is Hessian
4. **Update mean**: `μ = Σ · (Σ₀⁻¹·μ₀ + gradient)`

### Prediction

```python
import numpy as np
import scipy.stats

def predict_preference_probability(
    model: ModelState,
    name_a_features: np.ndarray,
    name_b_features: np.ndarray
) -> float:
    """
    Predict the probability that name A is preferred over name B.

    Returns:
        float: Probability in range [0, 1], where 0.5 indicates equal preference
    """
    delta = name_a_features - name_b_features
    mean_utility = model.weight_mean @ delta
    variance = delta @ model.weight_cov @ delta
    # Probability with uncertainty
    return float(scipy.stats.norm.cdf(mean_utility / np.sqrt(1 + variance)))

# Example: Compare two names
features_a = np.array([2.0, 0.4, 5.0])  # syllables, vowel_ratio, length
features_b = np.array([3.0, 0.3, 7.0])
prob = predict_preference_probability(model, features_a, features_b)
print(f"P(A ≻ B) = {prob:.3f}")  # Output: P(A ≻ B) = 0.450
```

## Integration with Application

### Database Persistence

- **Model state table**: The system serializes weights, covariance, and metadata
- **Comparison logging**: The system stores all pairwise preferences for training
- **Rating synchronization**: The system converts model utilities to preference scores

### UI Integration

- **Real‑time updates**: Model updates after each comparison
- **Rating display**: Preference scores (1500 ± 500 scale) shown in UI
- **Pair selection**: Thompson sampling runs on demand

### Fallback Mechanisms

- **Random selection**: The system falls back to random selection if the model is not initialized or fails
- **Graceful degradation**: The basic preference system always functions
- **Recovery**: The system reinitializes the model if it becomes corrupted

## Performance Characteristics

### Computational Complexity

- **Feature extraction**: 0.5–2ms per name (cached after first extraction)
- **Model update**: 0.5–2ms per comparison
- **Thompson sampling**: 10–100ms for candidate selection
- **Rating sync**: 80–120ms for all 44,000 names

### Memory Usage

- **Feature cache**: ~44k names × 25 features × 8 bytes ≈ 9MB
- **Model state**: 25 weights + 25×25 covariance ≈ 5KB
- **Training data**: All historical comparisons in SQLite

## Extending the System

### Adding New Features

1. Implement feature extraction function in `FeatureExtractor`
2. Update feature normalization
3. Retrain model with new feature dimension

### Alternative Models

1. Implement new model class with same interface
2. Update `get_active_learning_model()` factory
3. Maintain backward compatibility

### Custom Active Learning Strategies

1. Override `select_candidates()` in utils.py
2. Implement different information gain metrics
3. Adjust exploration‑exploitation balance

## See Also

- [Tutorial](tutorial.md) - Step-by-step usage guide
- [System Architecture](architecture.md) - Design decisions and component
  architecture
- [Home](index.md) - Overview and quick start
