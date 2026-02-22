---
icon: lucide/book-open
---

# Tutorial

Learn how to use the Name Ranking application in this step-by-step tutorial,
modeled after the excellent
[Typer tutorial](https://typer.tiangolo.com/tutorial/) structure.

This tutorial covers everything from the simplest installation to advanced
features, with clear explanations of how the application works and the theory
behind it.

## Quick Start

### The Simplest Example

Let's get the application running as quickly as possible. First, make sure you
have the prerequisites:

- **Python 3.13** or higher
- **Git** (for submodule management)
- **[uv](https://github.com/astral-sh/uv)** - fast Python package manager

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/yourusername/sort-names.git
cd sort-names

# Install dependencies
uv sync

# Initialize the database
uv run name-db init

# Start the application
uv run streamlit run src/st_name_ranking/main.py
```

The application will open in your default web browser at
`http://localhost:8501`.

> **Tip**: If you see "No names loaded in the database", click **Sync Names** in
> the sidebar. This loads names from the included dataset.

## First Steps

### What You'll See

When you first open the application, you'll see:

1. **Main Title**: "Name Preference Ranker"
2. **Sidebar**: Contains database management, filters, and controls
3. **Main Area**: Empty initially, will show tournament interface

### Syncing Names

The application needs names to work with. Click the **Sync Names** button in the
sidebar:

```
Sidebar → Database Management → Sync Names
```

You should see a toast notification: "Loaded X total names" where X is around
44,000 for the full Danish name dataset.

### Understanding the Interface

The application has two main tabs:

1. **Tournament**: Compare names side-by-side
2. **Similarity Search**: Find names similar to a target

> **Tip**: Names are loaded from a git submodule containing official Danish name
> data. The application tracks which names have been processed to avoid
> redundant work.

## Tournament Mode

### What is a Name Tournament?

The tournament interface shows two names side-by-side. Your job is simple:
choose which name you prefer, or mark them as equal.

Think of it like a sports tournament bracket, but for names!

### Your First Comparison

When you open the Tournament tab, you'll see:

```
Left Side: [Name A]
Right Side: [Name B]

Buttons: ← Prefer Left | Draw | Down | Prefer Right →
```

#### The Four Voting Options

1. **← Prefer Left**: You like the left name more than the right name
2. **Prefer Right →**: You like the right name more than the left name  
3. **Draw** (🤝): Both names are equally good (or equally acceptable)
4. **Down** (👎): **Dislike both names** - neither is appealing

> **Tip**: Use "Draw" when you're indifferent between two names you like, and "Down" when you actively dislike both options. The system learns differently from each choice.

**Try it now**: Click "Prefer Left" for the name on the left, or use the **left
arrow key** on your keyboard.

### What Happens Behind the Scenes

When you make a choice:

1. **Record Comparison**: Your preference is stored in the `comparisons` table with one of four values:
   - `-1`: Prefer left name (name_a > name_b)
   - `1`: Prefer right name (name_b > name_a)  
   - `0`: Draw (both equally preferred)
   - `2`: Down (dislike both names)
   
   > **Note**: The database schema automatically migrates to support the "down" preference for existing installations.

2. **Update Model**: The Bradley-Terry Bayesian model updates based on your preference:
   - For preferences `-1`/`1`: Updates relative strength between names
   - For draws (`0`): Treats names as equal in preference
   - For downs (`2`): Treats both names as less preferred than a neutral baseline

3. **Select New Pair**: Thompson sampling selects the next pair that maximizes information gain
4. **Show Progress**: The comparison counter increments and ratings update

```python
# Simplified version of what happens
def process_vote(name_a, name_b, preference):
    # Store comparison with preference value
    database.record_comparison(name_a, name_b, preference)
    
    # Update Bayesian model (handles all four preference types)
    model.update_based_on_preference(name_a, name_b, preference)
    
    # Sync ratings from updated model weights
    ratings = model.compute_ratings()
    database.update_ratings(ratings)
    
    # Select new names using active learning
    new_pair = select_next_names()
    return new_pair
```

### Keyboard Shortcuts

For rapid voting, use these keyboard shortcuts:

- **Left Arrow (←)**: Prefer the left name
- **Right Arrow (→)**: Prefer the right name  
- **Up Arrow (↑)**: Mark as a draw (both equally preferred)
- **Down Arrow (↓)**: Dislike both names
- **Space**: Show similarity between current names

> **Tip**: The system uses **active learning** to select names that will teach
> it the most about your preferences. Early on, it explores broadly; as it
> learns, it focuses on names you're likely to have strong opinions about.

## Understanding Preferences

### What is Bayesian Preference Learning?

The application doesn't just count votes—it learns a **mathematical model** of
your preferences. This model understands that names have features (phonetic,
linguistic, metadata) and that you might prefer certain types of names.

#### The Bradley-Terry Model

The core mathematical model is the **Bradley-Terry model**, which estimates the
probability that you'll prefer name A over name B:

```
P(A ≻ B) = σ(w·φ(A) - w·φ(B))
```

Where:

- `σ(x)` is the logistic function: `1 / (1 + exp(-x))`
- `φ(name)` is a **feature vector** representing the name
- `w` is a **weight vector** learned from your comparisons

#### Name Features

Each name is converted to a 25-dimensional feature vector including:

1. **Phonetic Features**: Double Metaphone codes (how the name sounds)
2. **Linguistic Features**: Syllable count, vowel ratio, length
3. **Metadata Features**: Gender, origin region, classification confidence

### Active Learning with Thompson Sampling

The system doesn't show random names. It uses **Thompson sampling** to balance:

- **Exploitation**: Show names where it's confident you have a preference
- **Exploration**: Show names it's uncertain about to learn more
- **Diversity**: Ensure you see different types of names

```python
# Simplified Thompson sampling
def select_next_names():
    # Sample weights from current Bayesian posterior
    sampled_weights = sample_from_posterior(model)

    # Compute preference scores for all names
    scores = compute_scores(sampled_weights, all_names)

    # Pick names that maximize information gain
    return select_informative_pair(scores, model.uncertainty)
```

> **Technical Detail**: The model uses **Laplace approximation** for efficient
> Bayesian updates. After each comparison, it updates a Gaussian posterior over
> the weight vector `w`.

## Similarity Search

### What is Similarity Search?

Sometimes you like a name and want to find similar ones. The similarity search
tab lets you find names similar to a target using three different methods.

### The Three Similarity Methods

#### 1. String Similarity (Levenshtein Distance)

- **What it does**: Measures edit distance between names
- **Good for**: Finding spelling variations
- **Example**: "Anna" → "Anne", "Anya", "Ana"

#### 2. Vector Similarity (LLM Embeddings)

- **What it does**: Uses semantic embeddings to find conceptually similar names
- **Good for**: Finding names with similar meanings or associations
- **Example**: "Knight" → "Warrior", "Guard", "Protector"

#### 3. Phonetic Similarity (Double Metaphone)

- **What it does**: Matches names that sound similar when spoken
- **Good for**: Finding names with similar pronunciation
- **Example**: "Smith" → "Smythe", "Schmidt", "Smit"

### Using Similarity Search

1. **Enter a target name** in the search box
2. **Select a similarity method** (string, vector, or phonetic)
3. **Adjust sliders** for number of results and similarity threshold
4. **Explore results** - click on similar names to see details

```python
# Example similarity search
similar = find_similar_names(
    target="Anna",
    method="phonetic",
    limit=10,
    threshold=0.7
)
# Returns: ["Anne", "Anya", "Ana", "Hanna", "Annika", ...]
```

### When to Use Each Method

- **Looking for variants**: Use **string similarity** (e.g., "Catherine" vs
  "Katherine")
- **Exploring themes**: Use **vector similarity** (e.g., "River" might match
  "Brook", "Lake")
- **Matching sound**: Use **phonetic similarity** (e.g., "Sean" vs "Shawn")

> **Tip**: The similarity search is particularly useful when you find a name you
> like and want to explore alternatives with similar characteristics.

## Origin Classification

### What is Origin Classification?

The application can automatically predict name nationalities using the
`ethnidata` library, then map countries to geographic regions.

### How It Works

1. **Batch Processing**: Names are classified in batches of 100
2. **Country Prediction**: `ethnidata` predicts the most likely nationality
3. **Region Mapping**: Countries are mapped to regions (Nordic, European, Asian,
   etc.)
4. **Confidence Scoring**: Each prediction includes a probability estimate
5. **Caching**: Results are stored in the database for future use

### Using Origin Classification

In the sidebar under "Database Management":

1. Click **Classify Origins**
2. Watch the progress bar as names are processed
3. Use the **Origin Filter** to explore names by region

```python
# Example classification
result = classify_name_origin("Lars")
# Returns: {
#   "country": "Denmark",
#   "region": "Nordic",
#   "confidence": 0.92
# }
```

### Geographic Regions

Names are mapped to these regions:

- **Nordic**: Denmark, Sweden, Norway, Finland, Iceland
- **European**: Germany, France, Italy, Spain, etc.
- **Asian**: China, Japan, Korea, India, etc.
- **Middle Eastern**: Arabic, Persian, Turkish names
- **African**: Names from various African countries
- **American**: English, Spanish, Portuguese names from the Americas

> **Tip**: Origin classification runs in the background. You can start it and
> continue voting while it processes. Classified names enable the origin filter
> in the sidebar.

## Advanced Features

### Filtering System

The sidebar includes filters to focus on specific types of names:

#### Gender Filter

- **All**: Show all names regardless of gender
- **Male**: Only masculine names
- **Female**: Only feminine names
- **Unisex**: Names used for both genders

#### Origin Filter

- Select one or more geographic regions
- Empty selection shows all regions
- Only available after origin classification

> **Tip**: Filters are persisted between sessions. Your filter settings are
> saved in the database.

### Rating Management

#### Save Ratings

Export your preferences as JSON for backup or analysis:

```
Sidebar → Export → Export Ratings as JSON
```

#### Reset Ratings

Start over with fresh preferences:

```
Sidebar → Ratings Management → Reset Ratings
```

**Warning**: This cannot be undone! All your comparison history will be lost.

#### Progress Tracking

The tournament interface shows:

- Total comparisons made
- Names compared in current session
- Top 10 ranked names based on current preferences

### Database Management (CLI)

For advanced users, there's a command-line interface:

```bash
# Initialize database (schema + sync names)
uv run name-db init

# Classify origins (100 names at a time)
uv run name-db process --limit 100

# Show database statistics
uv run name-db stats

# Check model status
uv run name-db model-status

# Reset the active learning model
uv run name-db model-reset
```

## Troubleshooting

### Common Issues and Solutions

#### "No names loaded in the database"

**Solution**: Click **Sync Names** in the sidebar. If that doesn't work:

```bash
# Ensure submodule is initialized
git submodule update --init

# Manually sync names
uv run name-db init
```

#### "Failed to classify origins"

**Solution**:

```bash
# Ensure ethnidata is installed
uv add ethnidata

# Check internet connection (required for model download)
# Try with a smaller batch
uv run name-db process --limit 10
```

#### "Model not updating"

**Solution**: The model needs data to learn!

- Make at least **10-20 comparisons** before expecting meaningful updates
- Try different types of names (use filters to explore)
- Reset the model if needed: `uv run name-db model-reset`

#### "Application running slowly"

**Solution**:

- Restart the application to clear caches
- Use filters to reduce the number of active names
- The first run is slowest (feature extraction cache warms up)

### Getting Help

- Check the [Active Learning System](active_learning.md) for theory details
- Review the [System Architecture](architecture.md) for technical understanding
- Look for error details in browser console (F12 → Console) or terminal logs
- The application logs to `streamlit.log` in the project directory

## Next Steps

Now that you understand the basics:

1. **Explore the Theory**: Read about the
   [Bayesian preference model](active_learning.md) in detail
2. **Understand the Architecture**: Learn about the
   [system design](architecture.md) and components
3. **Contribute**: Check the development guide for contributing to the project
4. **Extend**: Consider adding new features or customizing the model for your
   needs

Remember: The more comparisons you make, the better the model understands your
preferences. Happy naming! 🎉
