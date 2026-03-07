# Name Ranking Application

A **Streamlit** web application for ranking Danish names using **Bayesian
preference learning** with **active learning**.

## Try It Now

Get the application running in **3 steps**:

```bash
# Clone with submodules
$ git clone --recurse-submodules https://github.com/yourusername/sort-names.git
$ cd sort-names

# Install dependencies (requires Python >=3.12,<3.14)
$ uv sync

# Initialize database
$ uv run st-name-ranking db init
Database initialized successfully.
Synced 4,847 names from submodule.
```

Start the application:

```bash
$ uv run st-name-ranking serve

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

Open **http://localhost:8501** in your browser.

## What You Do

1. **Compare names**: Two names appear side-by-side
2. **Vote**: Click **← Prefer Left**, **Draw**, **Down**, or **Prefer Right →**
3. **Watch rankings**: Top 10 names update after each vote based on your
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
4. **Model updates**: Parameters update after each vote

After your first set of comparisons, the model starts adapting ranking and pair
selection to your preferences.

In the **Tournament** view, the sample-size selector defaults to the full
filtered set (**N**). Available options are **50, 100, 500, 1000, 2000, 3000,
..., N**.

The Tournament UI also shows queue refill health: **green/yellow/red** latency,
**last/avg refill ms**, and current queue fill.

## Documentation

### Start Here

- **[Tutorial](tutorial.md)** - Complete walkthrough from installation to daily
  usage

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

- **Python >=3.12,<3.14**
- **Git** (for submodule management)
- **[uv](https://github.com/astral-sh/uv)** package manager

## Acknowledgments

- [**ethnidata**](https://github.com/teyfikoz/ethnidata) for name nationality
  prediction
- **Streamlit** for the web application framework
- The Danish government for the name data
