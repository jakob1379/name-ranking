# Name Ranking

[![Docs](https://img.shields.io/badge/docs-live-2ea44f)](https://jakob1379.github.io/name-ranking)

Find the Danish names you actually like by answering one easy question at a
time: _this one or that one?_

<p align="center">
  <img src=".github/assets/preference-loop.svg" width="960" alt="The preference loop: you prefer Freja over Alma with the left arrow key, the Bayesian Bradley-Terry model's uncertainty narrows, Freja moves to the top of the ranking, and Thompson sampling picks the next pair, Ida and Karla">
</p>

## Why

The official list of approved Danish first names runs to thousands of entries.
Nobody can score that many names one by one, but anyone can say which of two
names they prefer.

Name Ranking is a local Streamlit app that turns those quick pairwise votes into
a full ranking. It describes every name by its features (phonetics, syllables,
letter patterns, gender, origin region), learns which features you favour with a
**Bayesian Bradley–Terry model**, and uses **Thompson sampling** to show you the
pair that teaches the model the most. Names you never voted on still get ranked.

## Quickstart

You need **Python 3.12 or 3.13**, **Git** and
[uv](https://github.com/astral-sh/uv).

```bash
# Clone with submodules (the name list lives in a Git submodule)
$ git clone --recurse-submodules https://github.com/jakob1379/name-ranking.git
$ cd name-ranking

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

## Using it

The **Tournament** view shows:

1. Two names side by side
2. Vote buttons: **← Prefer Left**, **Draw**, **Down**, **Prefer Right →**
3. Top 10 rankings that update after each vote
4. A **Tournament sample size** selector (50, 100, 500, 1000, 2000, 3000, ...,
   **N**)
5. A queue status line with **green/yellow/red** refill latency, **last/avg
   ms**, and queue fill

Vote faster with the arrow keys:

| Key   | Vote                |
| ----- | ------------------- |
| **←** | Prefer left         |
| **→** | Prefer right        |
| **↑** | Draw                |
| **↓** | Dislike both (Down) |

## Feature cache

`db init` pre-computes features for every name. Rebuild them after changing
feature engineering:

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

Other `db` commands: `stats`, `model status`, `model reset`, `origins classify`
and `import`.

## Learn more

- [Tutorial](docs/tutorial.md): a complete walkthrough
- [Active Learning System](docs/active_learning.md): the model, Thompson
  sampling and feature engineering
- [System Architecture](docs/architecture.md): components and data flow
- [Features](docs/features.md): everything the app can do

## Requirements

- **Python >=3.12,<3.14**
- **Git**
- [uv](https://github.com/astral-sh/uv) package manager
