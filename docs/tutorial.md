---
icon: lucide/book-open
---

# Tutorial

Learn how to use the Name Ranking application in this step-by-step tutorial.

This tutorial covers everything from installation to advanced features, with clear explanations of how the application works and the theory behind it.

## Installation

### Prerequisites

- Python 3.13 or higher
- Git (for submodule management)
- [uv](https://github.com/astral-sh/uv) - fast Python package manager

### Step 1: Clone the Repository

```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/yourusername/sort-names.git
cd sort-names
```

### Step 2: Install Dependencies

```bash
# Install all dependencies
uv sync
```

### Step 3: Initialize the Database

```bash
# Initialize database schema and sync names
uv run name-db init
```

### Step 4: (Optional) Classify Name Origins

```bash
# Process name nationalities using ethnidata
uv run name-db process --limit 100
```

## First Steps

### Starting the Application

```bash
# Launch the Streamlit application
uv run streamlit run src/st_name_ranking/main.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Initial Setup

1. **Sync Names**: On first run, click "Sync Names" in the sidebar to load names from the submodule
2. **Check Database**: Verify the database statistics show names loaded
3. **Apply Filters**: Use gender and origin filters to narrow down names

## Tournament Mode

The tournament interface lets you compare names pairwise and vote for your preferences.

### How It Works

1. **Pair Selection**: The system selects two names using Thompson sampling based on your previous preferences
2. **Voting Options**:
   - **Left Name** (← arrow key): Prefer the name on the left
   - **Right Name** (→ arrow key): Prefer the name on the right  
   - **Draw** (↓ arrow key): No preference between the names
3. **Real-time Updates**: Your preferences update a Bayesian model that learns across multiple dimensions

### Key Features

- **Active Learning**: Names are selected to maximize learning about your preferences
- **Keyboard Navigation**: Use arrow keys for rapid voting
- **Progress Tracking**: See how many comparisons you've made
- **Top Rankings**: View top-rated names based on learned preferences

### Example Workflow

```python
# Behind the scenes, each comparison updates the Bayesian model
def update_preference(winner, loser):
    # Extract features for both names
    features_a = extract_features(winner)
    features_b = extract_features(loser)
    
    # Update Bradley-Terry model with new comparison
    model.update(features_a, features_b, preference=1)
    
    # Sync ratings for display
    update_ratings_from_model()
```

## Similarity Search

The similarity search tab lets you find names similar to a target name using multiple methods.

### Search Methods

#### 1. String Similarity (Levenshtein Distance)
- Measures edit distance between names
- Good for finding spelling variations
- Example: "Anna" matches "Anne", "Anya"

#### 2. Vector Similarity (LLM Embeddings)
- Uses semantic embeddings to find conceptually similar names
- Captures meaning and associations
- Example: "Knight" might match "Warrior", "Guard"

#### 3. Phonetic Similarity (Double Metaphone)
- Matches names that sound similar
- Uses phonetic encoding algorithm
- Example: "Smith" matches "Smythe", "Schmidt"

### Using Similarity Search

1. **Enter Target Name**: Type a name in the search box
2. **Select Method**: Choose string, vector, or phonetic similarity
3. **Adjust Sliders**: Control the number of results and similarity threshold
4. **Explore Results**: Click on similar names to see details

### Example Query

```python
# Finding similar names programmatically
similar_names = find_similar_names(
    target="Anna",
    method="phonetic",
    limit=10,
    threshold=0.7
)
# Returns: ["Anne", "Anya", "Ana", "Hanna", ...]
```

## Origin Classification

The application can automatically predict name nationalities and map them to geographic regions.

### How It Works

1. **Batch Processing**: Names are classified in batches of 100 using the `ethnidata` library
2. **Geographic Mapping**: Countries are mapped to regions (Nordic, European, Asian, etc.)
3. **Confidence Scoring**: Each prediction includes a probability estimate
4. **Caching**: Results are stored in the database for future use

### Using Origin Classification

1. **Start Classification**: Click "Classify Origins" in the sidebar
2. **Monitor Progress**: Watch the progress bar as names are processed
3. **View Results**: Use the origin filter to explore names by region

### Example Classification

```python
# Classifying a single name
result = classify_name_origin("Lars")
# Returns: {"country": "Denmark", "region": "Nordic", "confidence": 0.92}
```

## Understanding Preferences

### Bayesian Preference Learning

The application uses a feature-based Bradley-Terry model to learn your preferences:

#### Mathematical Model

The probability that name A is preferred over name B is:

```
P(A ≻ B) = σ(w·φ(A) - w·φ(B))
```

Where:
- `σ(x) = 1 / (1 + exp(-x))` is the logistic function
- `φ(name)` is the 25-dimensional feature vector for a name
- `w` is the weight vector learned from your comparisons

#### Feature Engineering

Names are converted to feature vectors including:

1. **Phonetic Features**: Double Metaphone primary and secondary codes
2. **Linguistic Features**: Syllable count, vowel ratio, name length
3. **Metadata Features**: Gender, origin region, classification confidence

#### Bayesian Updates

After each comparison, the model updates using Laplace approximation:
- **Prior**: Gaussian distribution over weights
- **Posterior**: Updated based on observed preference
- **Uncertainty**: Covariance matrix tracks uncertainty in weights

### Active Learning with Thompson Sampling

The system uses Thompson sampling to balance:

- **Exploitation**: Compare names with high predicted preference difference
- **Exploration**: Compare names where the model is uncertain
- **Diversity**: Ensure coverage across different name characteristics

```python
# Thompson sampling algorithm
def select_candidates():
    # Sample weights from current posterior
    sampled_weights = sample_from_posterior(model)
    
    # Compute utilities for all names
    utilities = compute_utilities(sampled_weights, all_features)
    
    # Select names maximizing information gain
    return select_max_information_gain(utilities, model.uncertainty)
```

## Advanced Features

### Filtering System

- **Gender Filter**: Show only Male, Female, or Unisex names
- **Origin Filter**: Filter by geographic regions (Nordic, European, Asian, etc.)
- **Combined Filters**: Apply multiple filters simultaneously

### Rating Management

- **Save Ratings**: Export your preferences to JSON
- **Reset Ratings**: Start over with fresh preferences
- **Progress Tracking**: Monitor your comparison history

### Keyboard Shortcuts

- **Left Arrow (←)**: Prefer left name
- **Right Arrow (→)**: Prefer right name  
- **Down Arrow (↓)**: Mark as draw
- **Space**: Show similarity search for current names

### Database Management

```bash
# CLI commands for database management
uv run name-db init      # Initialize database
uv run name-db process   # Classify origins
uv run name-db stats     # Show statistics
uv run name-db model-status  # Check model status
```

## Troubleshooting

### Common Issues

#### "No names loaded in the database"
- Click "Sync Names" in the sidebar
- Ensure the git submodule is initialized (`git submodule update --init`)

#### "Failed to classify origins"
- Ensure `ethnidata` is installed (`uv add ethnidata`)
- Check internet connection for model downloads

#### "Model not updating"
- Ensure you have made several comparisons (model needs data)
- Try resetting the model: `uv run name-db model-reset`

#### "Application running slowly"
- Clear the feature cache by restarting the application
- Reduce the number of names being filtered

### Getting Help

- Check the [Active Learning System](active_learning.md) documentation
- Review the [System Architecture](architecture.md) for technical details
- Look for error details in the browser console or terminal logs

## Next Steps

Now that you understand the basics, you can:

1. **Explore the Theory**: Read about the [Bayesian preference model](active_learning.md)
2. **Understand the Architecture**: Learn about the [system design](architecture.md)
3. **Contribute**: Check the development guide for contributing to the project
4. **Extend**: Add new features or customize the model for your needs

---

*This tutorial is modeled after the excellent [Typer tutorial](https://typer.tiangolo.com/tutorial/) structure.*