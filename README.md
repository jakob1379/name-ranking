# Name Ranking

Rank Danish names using **Bayesian preference learning**.

## Quickstart

You need **Python 3.13+** and **Git**.

```bash
# Clone with submodules
$ git clone --recurse-submodules https://github.com/yourusername/sort-names.git
$ cd sort-names

# Install dependencies
$ uv sync

# Initialize database
$ uv run name-db init
Database initialized successfully.
Synced 4,847 names from submodule.

# Start the application
$ uv run streamlit run st_name_ranking/main.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
```

The application opens at **http://localhost:8501** in 2 seconds.

## What You See

1. Two names side-by-side
2. Vote buttons: **← Prefer Left**, **Draw**, **Down**, **Prefer Right →**
3. Top 10 rankings update in real-time

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

- **Python 3.13+**
- **Git**
- [uv](https://github.com/astral-sh/uv) package manager
