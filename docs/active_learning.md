# Active Learning System

The Name Ranking application uses a Bayesian preference learning system with
active learning to intelligently select name pairs for comparison. This system
replaces traditional ELO ratings with a feature‑based Bradley‑Terry model that
learns user preferences across multiple dimensions.

> **Note**: For a step-by-step usage guide, see the [Tutorial](tutorial.md). This document focuses on the theoretical foundations of the system.

## Overview

The active learning system has four main components:

1. **Feature Extraction**: Converts names into numerical feature vectors
2. **Bradley‑Terry Model**: Bayesian preference model with Laplace approximation
3. **Thompson Sampling**: Active learning algorithm for pair selection
4. **Model Persistence**: Database‑backed model state storage

## Feature Engineering

### Phonetic Features (Double Metaphone)

The **Double Metaphone** algorithm converts names to phonetic codes that capture
pronunciation similarities across languages:

- **Primary and secondary codes**: Each name produces two phonetic codes
- **Feature encoding**: Codes are one‑hot encoded as categorical features
- **Similarity computation**: Used for both pair selection and similarity search

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

We use a Gaussian prior on the weights with Laplace approximation for efficient
Bayesian updates:

- **Prior**: `w ∼ N(0, Σ₀)` with diagonal covariance `Σ₀ = I`
- **Posterior approximation**: Gaussian `N(μ, Σ)` updated after each comparison
- **Laplace approximation**: Second‑order Taylor expansion around MAP estimate

### Model State

```python
@dataclass
class ModelState:
    weight_mean: np.ndarray      # μ: mean weight vector (d,)
    weight_cov: np.ndarray       # Σ: covariance matrix (d, d)
    training_samples: int        # Number of training comparisons
    feature_names: list[str]     # Names of each feature dimension
```

## Thompson Sampling for Active Learning

### Exploration‑Exploitation Tradeoff

Thompson sampling balances:

- **Exploitation**: Compare names with high predicted preference difference
- **Exploration**: Compare names where the model is uncertain

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
def initialize_model(feature_dim: int) -> ModelState:
    return ModelState(
        weight_mean=np.zeros(feature_dim),
        weight_cov=np.eye(feature_dim),
        training_samples=0,
        feature_names=feature_names
    )
```

### Bayesian Update

After observing preference `y ∈ {-1, 0, 1}` for names A and B:

1. **Compute feature difference**: `Δφ = φ(A) - φ(B)`
2. **Compute MAP estimate**: Newton‑Raphson optimization
3. **Update covariance**: `Σ = (Σ₀⁻¹ + H)⁻¹` where H is Hessian
4. **Update mean**: `μ = Σ · (Σ₀⁻¹·μ₀ + gradient)`

### Prediction

```python
def predict_preference_probability(
    model: ModelState,
    name_a_features: np.ndarray,
    name_b_features: np.ndarray
) -> float:
    delta = name_a_features - name_b_features
    mean_utility = model.weight_mean @ delta
    variance = delta @ model.weight_cov @ delta
    # Probability with uncertainty
    return scipy.stats.norm.cdf(mean_utility / np.sqrt(1 + variance))
```

## Integration with Application

### Database Persistence

- **Model state table**: Serialized weights, covariance, and metadata
- **Comparison logging**: All pairwise preferences stored for training
- **Rating synchronization**: Model utilities converted to preference scores

### UI Integration

- **Real‑time updates**: Model updates after each comparison
- **Rating display**: Preference scores (1500 ± 500 scale) shown in UI
- **Pair selection**: Thompson sampling runs on demand

### Fallback Mechanisms

- **Random selection**: If model not initialized or fails
- **Graceful degradation**: Basic preference system always works
- **Recovery**: Model reinitialization if corrupted

## Performance Characteristics

### Computational Complexity

- **Feature extraction**: ~1ms per name (cached after first)
- **Model update**: ~1ms per comparison
- **Thompson sampling**: ~10‑100ms for candidate selection
- **Rating sync**: ~100ms for all 44k names

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
- [System Architecture](architecture.md) - Design decisions and component architecture
- [Home](index.md) - Overview and quick start
