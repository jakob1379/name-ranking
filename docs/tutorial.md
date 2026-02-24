# Tutorial

Learn to use the Name Ranking application through hands-on examples.

This tutorial takes you from installation to advanced usage in **15 minutes**.

## Prerequisites

You need:

- **Python 3.13+**
- **Git**
- [uv](https://github.com/astral-sh/uv) package manager

Verify your Python version:

```bash
$ python --version
Python 3.13.0
```

## Installation

### Step 1: Clone the Repository

```bash
# Clone with submodules to get the name dataset
$ git clone --recurse-submodules https://github.com/yourusername/sort-names.git
$ cd sort-names
```

### Step 2: Install Dependencies

```bash
# Install all packages
$ uv sync
Resolved 87 packages in 0.42s
Audited 87 packages in 0.01s
```

### Step 3: Initialize the Database

```bash
# Create database schema and load names
$ uv run st-name-ranking init
Database initialized successfully.
Synced 4,847 names from submodule.
```

The database now contains **4,847 Danish names**.

### Step 4: Start the Application

```bash
$ uv run st-name-ranking start

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

Your browser opens automatically. The application loads in **2 seconds**.

!!! note "No Automatic Sync" The application does not sync names automatically
on startup. You ran `st-name-ranking init` to load names via CLI. You can also click
**Sync Names** in the sidebar.

## First Comparison

### What You See

The **Tournament** tab displays:

1. **Two names** side-by-side (selected by **Thompson sampling**)
2. **Four voting buttons**:
   - **← Prefer Left**: You like the left name more
   - **Draw**: Both names are equal
   - **Down**: You dislike both names
   - **Prefer Right →**: You like the right name more
3. **Top 10 rankings**: Current best names based on your preferences
4. **Comparison counter**: How many votes you have made

### Make Your First Vote

Click **← Prefer Left** or press the **left arrow key**.

The application records your preference and displays two new names.

!!! tip "Keyboard Shortcuts" Use arrow keys for speed: - **←** Prefer left -
**→** Prefer right - **↑** Draw (equal) - **↓** Dislike both

### What Happens Behind the Scenes

When you vote, the system:

1. **Records the comparison** in **SQLite** with your preference (`-1`, `0`,
   `1`, or `2`)
2. **Updates the Bayesian model** using **Laplace approximation**
3. **Syncs ratings** from model weights to preference scores
4. **Selects new names** using **Thompson sampling** for maximum information
   gain

```python
from typing import Tuple

# Pseudo-code showing the voting process
def process_vote(name_a: str, name_b: str, preference: int) -> Tuple[str, str]:
    """Process a user vote and return the next name pair."""
    # Store comparison in database
    database.record_comparison(name_a, name_b, preference)

    # Update Bayesian model (Bradley-Terry with Laplace approximation)
    model.update_based_on_preference(name_a, name_b, preference)

    # Sync ratings from updated model weights
    ratings: dict = model.compute_ratings()
    database.update_ratings(ratings)

    # Select new names using Thompson sampling
    new_pair: Tuple[str, str] = select_next_names()
    return new_pair
```

## Understanding the Four Vote Types

The application learns differently from each vote type:

### Prefer Left (-1)

You like the left name more than the right.

The model learns: `preference(left) > preference(right)`

### Prefer Right (1)

You like the right name more than the left.

The model learns: `preference(right) > preference(left)`

### Draw (0)

You like both names equally (or dislike both equally).

The model learns: `preference(left) == preference(right)`

### Down (2)

You actively dislike both names.

The model learns: `preference(left) < neutral` and `preference(right) < neutral`

!!! warning "Down Votes" **Down** votes exclude names from positive rankings.
Use this when neither name appeals to you.

## How the Model Learns

The application does not count votes. It learns a **mathematical model** of your
preferences.

### Feature Extraction

Each name becomes a **25-dimensional feature vector**:

| Feature Category | Examples                                         |
| ---------------- | ------------------------------------------------ |
| **Phonetic**     | Double Metaphone codes (how the name sounds)     |
| **Linguistic**   | Syllable count, vowel ratio, name length         |
| **Metadata**     | Gender, origin region, classification confidence |

### Bradley-Terry Model

The model estimates the probability you prefer name A over name B:

```
P(A ≻ B) = σ(w·φ(A) - w·φ(B))
```

Where:

- `σ(x)` is the logistic function: `1 / (1 + exp(-x))`
- `φ(name)` is the **feature vector** for a name
- `w` is the **weight vector** learned from your comparisons

### Thompson Sampling

The system uses **Thompson sampling** to select name pairs:

1. **Sample weights** from the current Bayesian posterior: `w̃ ~ N(μ, Σ)`
2. **Compute utilities** for all names: `u(name) = w̃·φ(name)`
3. **Select informative pairs** that maximize expected information gain
4. **Balance exploration and exploitation**

After **20 comparisons**, the model understands your basic preferences. After
**50 comparisons**, it predicts your choices accurately.

## Similarity Search

Find names similar to one you like.

### Open Similarity Search

Click the **Similarity Search** tab.

### Enter a Target Name

Type a name in the search box, for example:

```
Anna
```

### Select a Method

Choose from three similarity methods:

#### 1. String Similarity (Levenshtein Distance)

Measures edit distance between names.

Good for: Finding spelling variations

Example: "Anna" → "Anne", "Anya", "Ana"

#### 2. Vector Similarity (LLM Embeddings)

Uses **semantic embeddings** for conceptual similarity.

Good for: Finding names with similar meanings

Example: "River" → "Brook", "Lake", "Stream"

#### 3. Phonetic Similarity (Double Metaphone)

Matches names that sound alike.

Good for: Finding similar pronunciations

Example: "Smith" → "Smythe", "Schmidt"

### View Results

Adjust the **number of results** and **similarity threshold** sliders.

The application displays similar names with similarity scores.

```python
from typing import List

# Example: Find phonetically similar names
similar: List[str] = find_similar_names(
    target="Anna",
    method="phonetic",
    limit=10,
    threshold=0.7
)
print(similar)
# Output: ["Anne", "Anya", "Ana", "Hanna", "Annika", ...]
```

## Origin Classification

Classify names by nationality and geographic region.

### Run Classification

In the sidebar under **Database Management**:

1. Click **Classify Origins**
2. The application processes **100 names at a time**
3. Watch the progress bar

Or use the CLI for batch processing:

```bash
# Classify 100 names
$ uv run st-name-ranking process --limit 100
Processing 100 names...
Classified: 97 Nordic, 3 European
Completed in 3.2s
```

### Geographic Regions

Names map to these regions:

- **Nordic**: Denmark, Sweden, Norway, Finland, Iceland
- **European**: Germany, France, Italy, Spain, etc.
- **Asian**: China, Japan, Korea, India, etc.
- **Middle Eastern**: Arabic, Persian, Turkish names
- **African**: Names from African countries
- **American**: English, Spanish, Portuguese names from the Americas

### Filter by Origin

After classification, use the **Origin Filter** in the sidebar:

1. Select one or more regions
2. The tournament shows only names from those regions
3. Rankings update to show filtered results

```python
from typing import Dict, Union

# Example classification result
result: Dict[str, Union[str, float]] = classify_name_origin("Lars")
print(result)
# Output:
# {
#   "country": "Denmark",
#   "region": "Nordic",
#   "confidence": 0.92
# }
```

## Filtering System

Focus on specific types of names.

### Gender Filter

In the sidebar:

- **All**: Show all names
- **Male**: Only masculine names
- **Female**: Only feminine names
- **Unisex**: Names used for both genders

### Origin Filter

Select one or more geographic regions.

Empty selection shows all regions.

!!! note "Filter Persistence" Filters persist between sessions. Your settings
save to the database.

## CLI Reference

The **Typer** CLI provides database management:

### Initialize Database

```bash
$ uv run st-name-ranking init
Database initialized successfully.
Synced 4,847 names from submodule.
```

### View Statistics

```bash
$ uv run st-name-ranking stats
Total names: 4,847
Classified: 4,623 (95.4%)
Comparisons: 1,247
```

### Check Model Status

```bash
$ uv run st-name-ranking model status
Model trained: Yes
Training samples: 1,247
Feature dimensions: 25
```

### Reset Model

```bash
$ uv run st-name-ranking model reset
Model state cleared successfully.
All ratings reset to default.
```

## Rating Management

### Export Ratings

Save your preferences as JSON:

1. Open the sidebar
2. Click **Export Ratings as JSON**
3. Save the file

### Reset Ratings

Start fresh:

1. Open the sidebar
2. Click **Reset Ratings**
3. Confirm the action

!!! warning "Reset is Permanent" Resetting deletes all comparison history. This
action cannot be undone.

## Troubleshooting

### "No names loaded in the database"

**Cause**: Names not synced from submodule.

**Solution**:

```bash
# Sync via CLI
$ uv run st-name-ranking init
Database initialized successfully.
Synced 4,847 names from submodule.
```

Or click **Sync Names** in the sidebar.

### "Failed to classify origins"

**Cause**: **ethnidata** library not installed or no internet connection.

**Solution**:

```bash
# Verify ethnidata is installed
$ uv add ethnidata

# Check internet connection (required for model download)
# Try with a smaller batch
$ uv run st-name-ranking process --limit 10
Processing 10 names...
Completed in 0.8s
```

### "Model not updating"

**Cause**: Not enough comparisons for meaningful learning.

**Solution**:

- Make at least **20 comparisons**
- Try different name types using filters
- Reset the model if needed: `uv run st-name-ranking model reset`

### "Application running slowly"

**Solution**:

- Restart the application to clear caches
- Use filters to reduce active names
- First run is slowest (feature cache warms up)

## Next Steps

You now understand:

1. **Installation and setup**
2. **Tournament voting** with four preference types
3. **Bayesian preference learning** and **Thompson sampling**
4. **Similarity search** with three methods
5. **Origin classification** and filtering
6. **CLI management** and troubleshooting

### Learn More

- [Active Learning System](active_learning.md) - Deep dive into **Bayesian
  preference modeling**
- [System Architecture](architecture.md) - Component design and data flow
- [Features](features.md) - Complete feature reference

### Practice

1. Make **50 comparisons** to train the model
2. Use **Similarity Search** to find variants of names you like
3. Run **Origin Classification** and filter by region
4. Export your ratings for backup

The more comparisons you make, the better the model understands your
preferences.
