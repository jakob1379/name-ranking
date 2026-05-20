# System Architecture

**Goal**: Understand how the Name Ranking application learns your preferences.

**Requirements**: **Python >=3.12,<3.14**

For usage instructions, see the [Tutorial](tutorial.md). This document explains
design decisions for developers.

## Try It Now

Explore the architecture in 3 steps:

```bash
# 1. Initialize the database
$ uv run st-name-ranking db init
Database initialized successfully.
Synced 4,847 names from submodule.

# 2. Check model status
$ uv run st-name-ranking db model status
Model trained: No
Training samples: 0
Feature dimensions: 25

# 3. View statistics
$ uv run st-name-ranking db stats
Total names: 4,847
Classified: 0 (0%)
Comparisons: 0
```

## Architecture Overview

```
┌─────────────────┐
│   UI Layer      │  Streamlit interface
│  (main.py)      │
└────────┬────────┘
         │
┌────────▼────────┐
│  Core Layer     │  Feature extraction, model, active learning
│ (features.py,   │
│  model.py)      │
└────────┬────────┘
         │
┌────────▼────────┐
│  Data Layer     │  SQLite database, Git submodule
│  (database.py)  │
└─────────────────┘
```

## Component Architecture

### Data Layer

#### SQLite Database

Stores 5 core tables:

1. **names**: Name metadata (name, gender, origin region)
2. **ratings**: Preference scores derived from Bayesian model
3. **comparisons**: Historical comparison data with 4 preference types:
   - `-1`: Prefer name_a over name_b
   - `1`: Prefer name_b over name_a
   - `0`: Draw (both equally preferred)
   - `2`: Down (dislike both names)
4. **model_state**: Active learning model parameters
5. **region_mapping**: Geographic region classifications

#### Git Submodule Integration

- **Commit hash tracking**: Avoids redundant processing
- **Incremental sync**: Processes only new or modified names
- **Data provenance**: Maintains lineage from source data

### Feature Extraction Layer (`features.py`)

#### FeatureExtractor Class

- **Phonetic encoding**: **Double Metaphone** generates primary/secondary codes
- **Linguistic analysis**: Syllable counting and vowel ratio calculation
- **Metadata encoding**: One-hot encoding of gender and origin region
- **Caching**: In-memory cache for 44,000+ names
- **Batch processing**: Feature extraction runs in batches for all names

#### Feature Pipeline

1. **Text normalization**: Lowercase and Unicode normalization
2. **Phonetic encoding**: Double Metaphone primary/secondary codes
3. **Linguistic analysis**: Character statistics and syllable counts
4. **Metadata lookup**: Database queries for gender and origin
5. **Normalization**: Min-max scaling to [0,1] range
6. **Vector assembly**: 25-dimensional feature vector

### Machine Learning Layer (`model.py`)

#### BradleyTerryModel Class

- **Bayesian inference**: Gaussian prior with **Laplace approximation**
- **Weight management**: 25-dimensional mean vector and 25×25 covariance matrix
- **Thompson sampling**: Balances exploration and exploitation
- **Database persistence**: Saves state to SQLite

#### Model Operations

- **Initialization**: Zero-mean prior with diagonal covariance
- **Update**: Bayesian update from each pairwise comparison
- **Prediction**: Computes preference probabilities between any 2 names
- **Sampling**: Thompson sampling for active learning
- **Persistence**: Serializes state to database

### Active Learning Layer (`active_learning/`)

#### Candidate Selection

- **Selection owner**: `active_learning/selection.py` owns feature lookup, model
  access, Thompson sampling, and fallback pair selection.
- **Background queue**: `active_learning/queue.py` keeps tournament pairs ready
  for responsive voting and reports refill latency.
- **Lazy updates**: `active_learning/lazy_updates.py` records comparisons,
  refreshes the model, and synchronizes model-derived ratings.
- **Compatibility facade**: `utils.py` only re-exports legacy helpers for older
  callers; new code should import active-learning services from this package.

#### Rating Synchronization

- **Utility computation**: Converts model weights to preference scores
- **Batch updates**: Processes all 44,000 names efficiently
- **Consistency checks**: Validates rating calculations

### User Interface Layer (`main.py`, `ui.py`)

- **Tournament interface**: Side-by-side name comparisons
- **Tournament sample-size selector**: Options are 50, 100, 500, 1000, 2000,
  3000, ..., N (defaults to full filtered set)
- **Queue refill latency indicator**: Green/yellow/red status with last/avg
  refill ms and queue fill level
- **Similarity search**: Multi-method name matching
- **Filter controls**: Gender and origin region filters
- **Administration**: Database sync and classification controls

### Command Line Interface (`cli.py`)

- `db init`: Initialize database (includes sync + feature extraction)
- `db stats`: Display database statistics
- `serve`: Launch the Streamlit web interface
- `db features rebuild`: Recompute all cached features
- `db features status`: Show feature cache status
- `db model status`: Show active learning model status
- `db model reset`: Reinitialize active learning model
- `db import`: Import database from file

## Data Flow

### Comparison Workflow

```python
from typing import Tuple

# 1. Select candidates using Thompson sampling
name_a, name_b = select_candidates()

# 2. User votes with one of 4 preferences
def process_vote(name_a: str, name_b: str, preference: int) -> None:
    # preference: -1 (prefer a), 1 (prefer b), 0 (draw), 2 (down)

    # 3. Record comparison in database
    database.record_comparison(name_a, name_b, preference)

    # 4. Update Bayesian model
    model.update_based_on_preference(name_a, name_b, preference)

    # 5. Sync ratings from model weights
    _update_ratings_from_model()
```

### Feature Computation Flow

```python
from typing import Dict, Any

# First-time feature extraction
def extract_all_features(names: list[str]) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for name in names:
        features[name] = feature_extractor.extract(name)
    return features

# Cached subsequent access
def get_name_features(name: str) -> Any:
    if name in cache:
        return cache[name]
    features = feature_extractor.extract(name)
    cache[name] = features
    return features
```

### Model Update Flow

```python
# 1. Record comparison with preference value
database.record_comparison(name_a_id, name_b_id, preference)

# 2. Extract features for both names
features_a = feature_extractor.extract(name_a)
features_b = feature_extractor.extract(name_b)

# 3. Update model (handles all 4 preference types)
model.update(features_a, features_b, preference)

# 4. Save state to database
model.save_to_db()

# 5. Update ratings for all names
ratings = model.compute_ratings()
database.update_ratings(ratings)
```

## Feature Caching System

The **Feature Caching System** stores pre-computed features for all names in the
database for repeated retrieval during model operations.

### Architecture

```
┌─────────────────────────────────────────────┐
│           Feature Cache Flow                │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐    ┌─────────────────┐    │
│  │  init cmd   │───▶│ Create Feature  │    │
│  │             │    │ Set (version)   │    │
│  └─────────────┘    └────────┬────────┘    │
│                              │             │
│                              ▼             │
│                     ┌─────────────────┐    │
│                     │ Batch Extract   │    │
│                     │ (100 names/batch)│   │
│                     └────────┬────────┘    │
│                              │             │
│                              ▼             │
│  ┌─────────────┐    ┌─────────────────┐    │
│  │  Model Use  │◀───│ Store in DB     │    │
│  │             │    │ name_features   │    │
│  └─────────────┘    └─────────────────┘    │
│                                             │
└─────────────────────────────────────────────┘
```

### Database Schema

#### name_features Table

```sql
CREATE TABLE name_features (
    name_id INTEGER NOT NULL REFERENCES names(id) ON DELETE CASCADE,
    feature_set_id INTEGER NOT NULL REFERENCES feature_sets(id) ON DELETE CASCADE,
    features_json TEXT NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (name_id, feature_set_id)
);
```

#### feature_sets Table

```sql
CREATE TABLE feature_sets (
    id INTEGER PRIMARY KEY,
    version TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_names_json TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 0
);
```

### Compute-Once Pattern

Features are extracted **once during initialization** and cached for reuse:

```python
# 1. During initialization - features computed for all names
$ uv run st-name-ranking db init
✓ Database schema created
✓ Synced 4,847 names from submodule
✓ Created feature set version: 20250224_120000
✓ Computed features for 4,847 names

# 2. During model operations - features retrieved from cache
features = get_cached_features(name_id)  # O(1) lookup
```

### Feature Set Versioning

Each feature extraction creates a new **version**:

- **Timestamp-based**: `YYYYMMDD_HHMMSS` format
- **Single active set**: Only one feature set is active at a time
- **Automatic cleanup**: Old feature sets remain for audit trails

**Rebuilding features** (after feature engineering changes):

```bash
# Recompute all features with new feature definitions
$ uv run st-name-ranking db features rebuild
✓ Cleared 4,847 cached features
✓ Created new feature set version: 20250224_130000
✓ Computed features for 4,847 names
```

### Storage Characteristics

| Aspect                | Value               |
| --------------------- | ------------------- |
| Storage per name      | ~500 bytes (JSON)   |
| Total for 4,847 names | ~2.4 MB             |
| Feature dimensions    | 25                  |
| Extraction time       | ~50ms per 100 names |

## Deployment Architecture

### Development Environment

- **Local SQLite**: Single-file database
- **Streamlit local server**: Development web server
- **UV package management**: Python dependency resolution

### Production Considerations

- **Database scaling**: SQLite supports single-user scenarios
- **Model persistence**: Stores state in database
- **Feature caching**: Pre-computed features in `name_features` table
- **Error resilience**: Graceful degradation on failures

## Performance Characteristics

### Memory Usage

| Component     | Size                                     |
| ------------- | ---------------------------------------- |
| Feature cache | ~9MB (44k names × 25 features × 8 bytes) |
| Model state   | ~5KB (25 weights + 25×25 covariance)     |
| Name data     | ~5-10MB                                  |

### Computation Time

| Operation          | Time                            |
| ------------------ | ------------------------------- |
| Feature extraction | ~1ms per name (cached)          |
| Model update       | ~1ms per comparison             |
| Pair selection     | ~10-100ms for Thompson sampling |
| Rating sync        | ~100ms for all 44k names        |

### Database Operations

| Operation            | Time               |
| -------------------- | ------------------ |
| Comparison recording | <1ms with indexes  |
| Rating updates       | ~100ms bulk update |
| Model persistence    | <10ms              |

## Security Considerations

### Data Protection

- **Local storage**: All data stays in local SQLite
- **No external APIs**: Origin classification uses local library
- **Input validation**: Sanitizes all name inputs
- **SQL injection prevention**: Uses parameterized queries

### User Privacy

- **No personal data**: Stores only name rankings
- **Anonymous usage**: No user accounts or tracking
- **Local processing**: All computation happens locally

## Extension Points

### New Feature Types

1. Add feature extraction function to `FeatureExtractor`
2. Update feature normalization pipeline
3. Retrain model with new feature dimension

### Alternative Models

1. Implement new model class with same interface
2. Update `get_active_learning_model()` factory function
3. Maintain backward compatibility

### UI Enhancements

1. Add new **Streamlit** components to `ui.py`
2. Extend filter options in sidebar
3. Add new visualization components

### Data Sources

1. Implement new data loader class
2. Add synchronization logic for new source
3. Update database schema as needed

## Design Principles

### Separation of Concerns

- **Data layer**: Pure database operations
- **ML layer**: Pure machine learning algorithms
- **UI layer**: Pure presentation logic
- **Integration layer**: Glue code with clear interfaces

### Backward Compatibility

- **API stability**: Maintains existing function signatures
- **Data migration**: Automatically upgrades schemas
- **UI consistency**: No breaking changes for users

### Progressive Enhancement

- **Basic functionality**: Preference system always works
- **Enhanced features**: Active learning when available
- **Graceful degradation**: Fallback mechanisms for failures

### Testability

- **Unit tests**: Isolated component testing
- **Integration tests**: Cross-component workflows
- **Mocking**: External dependencies mocked for tests

## See Also

- [Tutorial](tutorial.md) - Step-by-step usage guide
- [Active Learning System](active_learning.md) - **Bayesian preference
  modeling** theory
- [Features](features.md) - Complete feature list
