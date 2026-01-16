---
icon: lucide/rocket
---

# Name Ranking Application

A Streamlit-based web application for ranking Danish names using Bayesian
preference learning with active learning, with similarity search, origin
classification, and intelligent pair selection.

## Features

### Name Ranking Tournament

- Compare two names selected by active learning (Thompson sampling)
- Vote for preferred name or mark as draw
- Real-time Bayesian preference updates with comparison tracking
- Top 10 rankings display based on learned preferences
- Keyboard shortcuts (arrow keys)

### Multi-method Similarity Search

- **String similarity** (Levenshtein distance) for name matching
- **Vector similarity** (LLM embeddings) for semantic matching
- **Phonetic similarity** (Double Metaphone) for phonetic name matching
- Find names similar to a reference name using multiple criteria

### Origin Classification

- **Optional classification** - runs only when explicitly requested
- **Incremental processing** - classify 100 names at a time or all at once
- Automatic nationality prediction using `ethnidata`
- Mapping to geographic regions (Nordic, European, Asian, etc.)
- Confidence scoring for predictions

### Intelligent Pair Selection

- **Comparison tracking** - records every pairwise vote to understand name
  popularity
- **Phonetic similarity** - uses Double Metaphone algorithm to find phonetically
  similar names
- **Smart candidate selection** - prioritizes names with fewer comparisons and
  phonetically interesting pairs
- **Automatic metadata collection** - builds a feature dataset for Bayesian
  preference learning

### Filtering & Management

- Gender filtering (Male, Female, Unisex, All)
- Origin region filtering (Nordic, European, International, etc.)
- Database-backed name storage with SQLite
- Git submodule integration for name data updates
- Ratings persistence with automatic comparison logging

## Quick Start

### Prerequisites

- Python 3.13 or higher
- Git (for submodule management)
- [uv](https://github.com/astral-sh/uv) - fast Python package manager

### Installation & Setup

```bash
# Clone repository with submodules
git clone --recurse-submodules https://github.com/yourusername/sort-names.git
cd sort-names

# Install dependencies
uv sync

# Initialize database
uv run name-db init

# Start the application
uv run streamlit run st_name_ranking/main.py
```

The application will be available at `http://localhost:8501`.

## Documentation

### Overview Documentation

- [README.md](../README.md) - Complete project overview, features, and
  quickstart guide

### Technical Documentation

- [Active Learning System](active_learning.md) - Bayesian preference modeling,
  feature engineering, and Thompson sampling
- [System Architecture](architecture.md) - Component architecture, data flow,
  and design principles

### Reference Documentation

- **Database Schema**: Complete SQL schema documentation
- **CLI Reference**: Command-line interface usage and options
- **API Reference**: Python module interfaces and functions
- **Testing Guide**: Test suite organization and execution

### Development Resources

- **Project Structure**: Module organization and responsibilities
- **Code Quality**: Linting, formatting, and type checking
- **Deployment**: Local development and production considerations

## Development

### Project Structure

See the [README.md](../README.md) for the complete project structure and module
descriptions.

### Testing

```bash
# Run all non-UI tests
uv run pytest -m "not playwright"

# Run specific test modules
uv run pytest tests/test_database.py
uv run pytest tests/test_utils.py
```

### Code Quality

- Uses `ruff` for linting and formatting
- Follows Python type hints
- Consistent code formatting with `ruff format`
