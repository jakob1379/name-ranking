# Name Ranking Application

A **Streamlit** web application for ranking Danish names using **Bayesian
preference learning** with **active learning**.

## Try It Now

Get the application running in **3 steps**:

```bash
# Clone with submodules
$ git clone --recurse-submodules https://github.com/yourusername/sort-names.git
$ cd sort-names

# Install dependencies (requires Python 3.13+)
$ uv sync

# Initialize database
$ uv run name-db init
Database initialized successfully.
Synced 4,847 names from submodule.
```

Start the application:

```bash
$ uv run streamlit run src/st_name_ranking/main.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

The application opens at **http://localhost:8501** in 2 seconds.

## What You Do

1. **Compare names**: Two names appear side-by-side
2. **Vote**: Click **← Prefer Left**, **Draw**, **Down**, or **Prefer Right →**
3. **Watch rankings**: Top 10 names update in real-time based on your
   preferences

Use **arrow keys** for speed:

- **←** Prefer left name
- **→** Prefer right name
- **↑** Draw (equal preference)
- **↓** Dislike both names

## How It Works

The application uses **Bayesian preference learning** to understand your taste:

1. **Feature extraction**: Converts each name into a 25-dimensional feature
   vector (**phonetic**, **linguistic**, **metadata**)
2. **Bradley-Terry model**: Learns a weight vector representing your preferences
3. **Thompson sampling**: Selects the most informative name pairs for comparison
4. **Real-time updates**: Model updates after each vote

After **20 comparisons**, the system understands your preferences well. After
**50 comparisons**, it predicts your choices with high accuracy.

## Documentation

### Start Here

- **[Tutorial](tutorial.md)** - Complete walkthrough from installation to
  advanced usage

### Learn the System

- **[Active Learning System](active_learning.md)** - **Bayesian preference
  modeling**, **Thompson sampling**, and feature engineering
- **[System Architecture](architecture.md)** - Component design, data flow, and
  implementation details

### Reference

- **[Features](features.md)** - Complete feature list and capabilities

## Development

Run tests:

```bash
# Run all non-UI tests
$ uv run pytest -m "not playwright"
====================== test session starts ======================
platform linux -- Python 3.13.0
collected 47 items

47 passed in 3.24s
```

Format and lint:

```bash
$ uv run ruff format .
84 files reformatted

$ uv run ruff check .
All checks passed
```

## Requirements

- **Python 3.13+** (required for modern type hints and features)
- **Git** (for submodule management)
- **[uv](https://github.com/astral-sh/uv)** package manager

## Acknowledgments

- [**ethnidata**](https://github.com/teyfikoz/ethnidata) for name nationality
  prediction
- **Streamlit** for the web application framework
- The Danish government for the name data
