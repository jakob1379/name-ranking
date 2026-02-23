---
icon: lucide/rocket
---

# Name Ranking Application

A Streamlit-based web application for ranking Danish names using **Bayesian
preference learning** with **active learning**, **similarity search**, and **origin
classification**.

## Try It Now

Get the application running in **5 minutes**:

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/yourusername/sort-names.git
cd sort-names

# Install dependencies (requires Python 3.13+)
uv sync

# Initialize database
uv run name-db init

# Start the application
uv run streamlit run st_name_ranking/main.py
```

Open http://localhost:8501 in your browser and start ranking names immediately.

## Overview

The Name Ranking application helps users discover and rank Danish names through
an interactive tournament-style interface. The system uses machine learning to
intelligently select name pairs for comparison, learns your preferences across
20-50 comparisons, and provides multiple ways to explore names through similarity
search and origin classification.

### Key Features

- **🎯 Active Learning Tournament**: Compare names selected by **Thompson
  sampling** for optimal preference learning
- **🔍 Multi‑Method Similarity Search**: Find similar names using **string
  similarity**, **semantic embeddings**, and **phonetic matching**
- **🌍 Origin Classification**: Automatic nationality prediction with **geographic
  region mapping**
- **📊 Bayesian Preference Modeling**: **Feature‑based Bradley‑Terry model**
  replaces traditional ELO ratings
- **🗃️ Database‑Backed**: **SQLite** storage with **git submodule** integration
  for name data
- **⌨️ Keyboard‑First Interface**: **Arrow key** navigation for rapid voting

## Quick Start

### Prerequisites

- **Python 3.13+** (required)
- **Git** (for submodule management)
- **[uv](https://github.com/astral-sh/uv)** - fast Python package manager

### Installation

```bash
# Clone repository with submodules
$ git clone --recurse-submodules https://github.com/yourusername/sort-names.git
$ cd sort-names

# Install dependencies
$ uv sync

# Initialize database (schema + sync names)
$ uv run name-db init
Database initialized successfully.
Synced 2473 names.

# Optional: Classify name origins (adds 2-3 minutes)
$ uv run name-db process
Processing complete: 2473 names classified.

# Start the application
$ uv run streamlit run st_name_ranking/main.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

The application opens at **http://localhost:8501**.

### First Run

1. **Sync Names**: Click **"Sync Names"** in the sidebar to load names from the
   submodule
2. **Classify Origins** (Optional): Click **"Classify Origins"** to process name
   nationalities
3. **Start Ranking**: Use **arrow keys** or click buttons to vote on name pairs
4. **Explore Similar Names**: Use the **similarity search tab** to find related
   names

## Documentation

### Tutorial

- [Step-by-Step Tutorial](tutorial.md) - Complete walkthrough from installation
  to advanced usage, following the
  [Typer tutorial](https://typer.tiangolo.com/tutorial/) style with progressive
  examples and clear explanations

### Theory & Architecture

- [Active Learning System](active_learning.md) - Bayesian preference modeling,
  feature engineering, and Thompson sampling
- [System Architecture](architecture.md) - Component architecture, data flow,
  and design principles

### User Guides

_All covered in the [Tutorial](tutorial.md):_

- **Getting Started** - Quick installation and first run (see
  [Quick Start](tutorial.md#quick-start))
- **Tournament Mode** - How to rank names effectively (see
  [Tournament Mode](tutorial.md#tournament-mode))
- **Similarity Search** - Finding names by multiple criteria (see
  [Similarity Search](tutorial.md#similarity-search))
- **Origin Classification** - Understanding name nationalities (see
  [Origin Classification](tutorial.md#origin-classification))
- **Filtering & Preferences** - Using filters and managing preferences (see
  [Advanced Features](tutorial.md#advanced-features))

### Technical Reference

- **Database Schema** - Complete SQL schema documentation
- **CLI Reference** - Command‑line interface usage and options
- **API Reference** - Python module interfaces and functions
- **Configuration** - Environment variables and settings

### Development

- **Project Structure** - Module organization and responsibilities
- **Testing Guide** - Test suite organization and execution
- **Code Quality** - Linting, formatting, and type checking
- **Deployment** - Local development and production considerations
- **Contributing** - How to contribute to the project

## Features in Detail

### Name Ranking Tournament

Compare two names selected by **active learning** (**Thompson sampling**). Vote
for your preferred name or mark as a draw. The system learns your preferences
across 20-50 comparisons and displays top rankings based on **Bayesian
preference scores**.

### Similarity Search

- **String Similarity**: **Levenshtein distance** for name matching
- **Vector Similarity**: **LLM embeddings** for semantic matching
- **Phonetic Similarity**: **Double Metaphone** algorithm for phonetic matching

### Origin Classification

- **Batch Processing**: Classify 100 names at a time or all 2,473 names at once
- **Geographic Regions**: Maps names to **Nordic**, **European**, **Asian**, and
  other regions
- **Confidence Scoring**: Displays probability estimates for each prediction

### Intelligent Pair Selection

- **Comparison Tracking**: Records every pairwise vote in the **SQLite**
  database
- **Phonetic Analysis**: Identifies phonetically similar names for comparison
- **Feature‑Based Learning**: Uses **phonetic**, **linguistic**, and **metadata**
  features

## Development

### Project Structure

See the [README.md](../README.md) for the complete project structure and module
descriptions.

### Testing

```bash
# Run all non‑UI tests
$ uv run pytest -m "not playwright"
====================== test session starts ======================
collected 47 items

src/st_name_ranking/tests/test_database.py .............
src/st_name_ranking/tests/test_utils.py .................

====================== 47 passed in 3.2s =======================

# Run with coverage reporting
$ uv run pytest --cov=st_name_ranking --cov-report=html
====================== test session starts ======================
collected 47 items

---------- coverage: platform linux, python 3.13.0 ----------
Name                          Stmts   Miss  Cover
-----------------------------------------------
...
TOTAL                           892     45    95%

# Run specific test modules
$ uv run pytest tests/test_database.py tests/test_utils.py
====================== 2 passed in 0.8s =======================
```

### Code Quality

```bash
# Run all pre-commit hooks
$ uv run prek run -a
ruff.....................................................................Passed
ruff-format..............................................................Passed
mypy.....................................................................Passed
pre-commit hook(s) made no changes.
```

### Database Management

```bash
# Initialize database
$ uv run name-db init
Database initialized successfully.
Synced 2473 names.

# Classify origins (100 names)
$ uv run name-db process --limit 100
Processing 100 names...
Done: 100 classified in 45s.

# Show statistics
$ uv run name-db stats
Total names: 2473
Classified:  2145 (87%)
Comparisons: 342

# Check model status
$ uv run name-db model-status
Model exists: Yes
Last updated: 2024-01-15 14:32:11
Comparisons:  342
Features:     12
```

## License

[Add your license here]

## Acknowledgments

- [ethnidata](https://github.com/teyfikoz/ethnidata) for name nationality
  prediction
- Streamlit for the web application framework
- The Danish government for the name data
