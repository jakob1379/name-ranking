# Name Ranking

[![Docs](https://img.shields.io/badge/docs-live-2ea44f)](https://jakob1379.github.io/name-ranking)

Rank Danish names using **Bayesian preference learning**.

## Quickstart

You need **Python 3.12 or 3.13** and **Git**.

```bash
# Clone with submodules
$ git clone --recurse-submodules https://github.com/yourusername/sort-names.git
$ cd sort-names

# Install dependencies
$ uv sync

# Optional: enable the ethnicolr fallback classifier
$ uv sync --extra origin-classification

# Initialize database (computes features automatically)
$ uv run st-name-ranking db init
✓ Database schema created
✓ Synced 4,847 names from submodule
✓ Created feature set version: 20250224_120000
✓ Computed features for 4,847 names

# Start the application
$ uv run st-name-ranking serve

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

Open **http://localhost:8501** in your browser.

### Feature Cache Management

The application **pre-computes features** for all names during initialization:

```bash
# Rebuild features after feature engineering changes
$ uv run st-name-ranking db features rebuild
✓ Cleared 4,847 cached features
✓ Created new feature set version: 20250224_130000
✓ Computed features for 4,847 names

# Check feature cache status
$ uv run st-name-ranking db features status
Names with features: 4,847 (100.0%)
Feature sets: 1
Active version: 20250224_120000
```

## What You See

1. Two names side-by-side
2. Vote buttons: **← Prefer Left**, **Draw**, **Down**, **Prefer Right →**
3. Top 10 rankings update after each vote
4. A **Tournament sample size** selector (50, 100, 500, 1000, 2000, 3000, ...,
   **N**)
5. A queue status line with **green/yellow/red** refill latency, **last/avg
   ms**, and queue fill

Use **arrow keys** for rapid voting:

- **←** Prefer left
- **→** Prefer right
- **↑** Draw
- **↓** Dislike both

## Next Steps

- Read the [Tutorial](docs/tutorial.md) for a complete walkthrough
- Learn about the [Active Learning System](docs/active_learning.md)
- See [System Architecture](docs/architecture.md) for technical details
- View all [Features](docs/features.md)

## Requirements

- **Python >=3.12,<3.14**
- **Git**
- [uv](https://github.com/astral-sh/uv) package manager
