# Features

Complete feature list for the Name Ranking application.

## Name Ranking Tournament

Compare names using **active learning** to learn your preferences efficiently.

- **Pair selection via Thompson sampling**: The system selects maximally
  informative name pairs
- **Four voting options**: Prefer left, prefer right, draw, or dislike both
- **Real-time Bayesian updates**: Model updates after each comparison
- **Top 10 rankings**: Current best names based on learned preferences
- **Keyboard shortcuts**: Arrow keys for rapid voting

## Similarity Search

Find names similar to a reference name using three methods:

### String Similarity (Levenshtein Distance)

Measures edit distance between names.

- Use for: Finding spelling variations
- Example: "Anna" → "Anne", "Anya", "Ana"

### Vector Similarity (LLM Embeddings)

Uses **semantic embeddings** for conceptual matching.

- Use for: Finding names with similar meanings
- Example: "Knight" → "Warrior", "Guard"

### Phonetic Similarity (Double Metaphone)

Matches names that sound alike when spoken.

- Use for: Finding similar pronunciations
- Example: "Smith" → "Smythe", "Schmidt"

## Origin Classification

Predict name nationalities using **ethnidata**.

- **Optional processing**: Run only when explicitly requested
- **Batch processing**: Classify 100 names at a time or all at once
- **Geographic region mapping**: Maps countries to regions (Nordic, European,
  Asian, etc.)
- **Confidence scoring**: Probability estimates for each prediction
- **Progress tracking**: Shows classification percentage in UI

### Geographic Regions

- **Nordic**: Denmark, Sweden, Norway, Finland, Iceland
- **European**: Germany, France, Italy, Spain, etc.
- **Asian**: China, Japan, Korea, India, etc.
- **Middle Eastern**: Arabic, Persian, Turkish names
- **African**: Names from African countries
- **American**: English, Spanish, Portuguese names from the Americas

## Filtering and Management

### Gender Filter

- **All**: Show all names regardless of gender
- **Male**: Only masculine names
- **Female**: Only feminine names
- **Unisex**: Names used for both genders

### Origin Filter

Select one or more geographic regions to filter names.

### Database Management

- **Sync Names**: Update database from **Git** submodule
- **Classify Origins**: Process name nationalities in batches
- **Export Ratings**: Save preferences as JSON
- **Reset Ratings**: Clear all comparison history

## Active Learning System

### Bayesian Preference Modeling

- **Feature-based Bradley-Terry model**: Learns preferences from name features
- **Laplace approximation**: Efficient Bayesian updates
- **25-dimensional feature vectors**: Phonetic, linguistic, and metadata
  features
- **Covariance matrix**: Models uncertainty in preferences

### Feature Extraction

Each name converts to a feature vector including:

1. **Phonetic features**: **Double Metaphone** primary/secondary codes
2. **Linguistic features**: Syllable count, vowel ratio, name length
3. **Metadata features**: Gender, origin region, classification confidence

### Thompson Sampling

- **Exploration-exploitation balance**: Learns efficiently with minimal
  comparisons
- **Information gain maximization**: Selects pairs that teach the most
- **Diversity constraint**: Ensures coverage across feature space

## Pre-computed Features

The application uses a **feature caching system** to store extracted features
for all names in the database. This eliminates redundant computation during
model training and inference.

### Feature Categories

| Category       | Features   | Description                                                    |
| -------------- | ---------- | -------------------------------------------------------------- |
| **Phonetic**   | 6 features | **Double Metaphone** encoding (position codes, length, vowels) |
| **Linguistic** | 9 features | Syllable count, vowel/consonant ratios, Danish letters         |
| **Metadata**   | 9 features | Gender encoding, origin region encoding                        |
| **Position**   | 1 feature  | First/last letter encoding                                     |

### Feature Extraction Flow

```bash
# 1. Initialize database - features computed automatically
$ uv run st-name-ranking init
✓ Database schema created
✓ Synced 4,847 names from submodule
✓ Created feature set version: 20250224_120000
✓ Computed features for 4,847 names

# 2. Check feature cache status
$ uv run st-name-ranking features status
Names with features: 4,847 (100.0%)
Feature sets: 1
Active version: 20250224_120000
✓ All names have cached features
```

### Feature Versioning

Features are versioned to support schema evolution:

- **Version format**: `YYYYMMDD_HHMMSS` (timestamp)
- **Active set**: Only one feature set is marked active
- **Storage**: Features stored as JSON in `name_features` table

### Adding New Features

To add new features to the extraction pipeline:

1. **Add extraction function** in `features.py`:

```python
def extract_custom_features(name: str) -> dict[str, float]:
    """Extract custom features for a name."""
    return {
        "custom_feature_1": compute_value_1(name),
        "custom_feature_2": compute_value_2(name),
    }
```

2. **Integrate into pipeline**:

```python
def extract_all_features(name, gender, origin_region,
                         include_custom=True):  # Add flag
    features = {}
    # ... existing features ...
    if include_custom:
        features.update(extract_custom_features(name))
    return features, feature_names
```

3. **Rebuild feature cache**:

```bash
$ uv run st-name-ranking features rebuild
✓ Cleared 4,847 cached features
✓ Created new feature set version: 20250224_130000
✓ Computed features for 4,847 names
```

### Feature Cache Commands

| Command                           | Purpose                                      |
| --------------------------------- | -------------------------------------------- |
| `st-name-ranking init`            | Initialize database and compute features     |
| `st-name-ranking features rebuild`| Recompute all features                       |
| `st-name-ranking features status` | Show cache statistics                        |
| `st-name-ranking start`           | Launch the Streamlit web interface           |

## Performance Optimizations

- **2-second startup**: No automatic sync on app launch
- **Feature caching**: In-memory cache for 44,000+ names
- **Batch processing**: Reduces **ethnidata** API calls by 100x
- **Efficient database sync**: Commit hash tracking avoids redundant processing
- **Bulk inserts**: Fast database operations with `executemany`
- **Vectorized computations**: **NumPy**-optimized pair scoring
- **Candidate queue**: Pre-fetches name pairs to eliminate selection latency

## Command Line Interface

The **Typer** CLI provides database management:

```bash
# Initialize database (computes features by default)
$ uv run st-name-ranking init

# Start the Streamlit web interface
$ uv run st-name-ranking start

# Show statistics
$ uv run st-name-ranking stats

# Feature cache management
$ uv run st-name-ranking features rebuild    # Recompute all features
$ uv run st-name-ranking features status     # Show cache status

# Model management
$ uv run st-name-ranking model status        # Check model status
$ uv run st-name-ranking model reset         # Reset active learning model

# Import database
$ uv run st-name-ranking import <file.db>
```

## Keyboard Shortcuts

| Key       | Action                                |
| --------- | ------------------------------------- |
| **←**     | Prefer left name                      |
| **→**     | Prefer right name                     |
| **↑**     | Draw (equal preference)               |
| **↓**     | Dislike both names                    |
| **Space** | Show similarity between current names |

## Database Schema

The application uses **SQLite** with these tables:

### names

```sql
CREATE TABLE names (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    gender TEXT CHECK(gender IN ('Male', 'Female', 'Unisex')),
    origin_region TEXT,
    origin_confidence REAL,
    origin_classified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### ratings

```sql
CREATE TABLE ratings (
    name_id INTEGER PRIMARY KEY REFERENCES names(id) ON DELETE CASCADE,
    rating REAL NOT NULL DEFAULT 1500.0,
    matches INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### comparisons

```sql
CREATE TABLE comparisons (
    id INTEGER PRIMARY KEY,
    name_a_id INTEGER NOT NULL REFERENCES names(id),
    name_b_id INTEGER NOT NULL REFERENCES names(id),
    preference INTEGER NOT NULL CHECK(preference IN (-1, 0, 1, 2)),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name_a_id, name_b_id, preference)
);
```

### model_state

```sql
CREATE TABLE model_state (
    id INTEGER PRIMARY KEY,
    mean_weights BLOB NOT NULL,
    covariance BLOB NOT NULL,
    feature_names BLOB NOT NULL,
    feature_dim INTEGER NOT NULL,
    training_samples INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Technical Specifications

### Requirements

- **Python 3.13+** (required for modern type hints)
- **Git** (for submodule management)
- **[uv](https://github.com/astral-sh/uv)** package manager

### Performance Characteristics

| Operation           | Time                   |
| ------------------- | ---------------------- |
| Application startup | 2 seconds              |
| Feature extraction  | 1ms per name (cached)  |
| Model update        | 1ms per comparison     |
| Thompson sampling   | 10-100ms               |
| Rating sync         | 100ms for 44,000 names |

### Memory Usage

| Component     | Size                                 |
| ------------- | ------------------------------------ |
| Feature cache | ~9MB (44k names × 25 features)       |
| Model state   | ~5KB (25 weights + 25×25 covariance) |
| Name data     | ~5-10MB                              |

## See Also

- [Tutorial](tutorial.md) - Step-by-step usage guide
- [Active Learning System](active_learning.md) - **Bayesian preference
  modeling** details
- [System Architecture](architecture.md) - Component design and data flow
