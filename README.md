# Name Ranking Application

Rank Danish names using **Bayesian preference learning** with **active learning**, **similarity search**, and **origin classification**.

## Try It Now

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/yourusername/sort-names.git
cd sort-names

# Install dependencies (takes 30 seconds)
uv sync

# Initialize database
uv run name-db init

# Launch the application
uv run streamlit run st_name_ranking/main.py
```

Open `http://localhost:8501` in your browser. The application starts in 2 seconds.

## Table of Contents

- [Features](#features)
  - [Name Ranking Tournament](#name-ranking-tournament)
  - [Similarity Search](#similarity-search)
  - [Origin Classification](#origin-classification)
  - [Filtering & Management](#filtering--management)
  - [Intelligent Pair Selection](#intelligent-pair-selection)
  - [Active Learning with Bayesian Preference Modeling](#active-learning-with-bayesian-preference-modeling)
  - [Performance Optimizations](#performance-optimizations)
- [Quickstart](#quickstart)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Setup Details](#setup-details)
- [Basic Usage](#basic-usage)
- [Development](#development)
  - [Project Structure](#project-structure)
  - [Testing](#testing)
  - [Code Quality](#code-quality)
  - [Generating Screenshots](#generating-screenshots)
- [Database Schema](#database-schema)
- [Origin Classification](#origin-classification-1)
- [Detailed Usage](#detailed-usage)
  - [CLI Reference](#cli-reference)
  - [Web Application Workflow](#web-application-workflow)
- [Performance Optimizations](#performance-optimizations-1)
- [Documentation](#documentation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

### Name Ranking Tournament

- Compare two names selected by active learning (Thompson sampling)
- Vote for preferred name or mark as draw
- Real-time Bayesian preference updates
- Top 10 rankings display based on learned preferences
- Keyboard shortcuts (**arrow keys**)

### Similarity Search

- **String similarity** (**Levenshtein distance**) for name matching
- **Vector similarity** (**LLM embeddings**) for semantic matching
- **Phonetic similarity** (**Double Metaphone**) for phonetic name matching
- Find names similar to a reference name using multiple criteria

### Origin Classification

- **Optional classification** - runs only when explicitly requested
- **Incremental processing** - classify 100 names at a time or all at once
- Automatic nationality prediction using **ethnidata**
- Mapping to geographic regions (Nordic, European, Asian, etc.)
- Confidence scoring for predictions
- Batch processing for unclassified names
- **Progress tracking** - shows classification percentage in UI

### Filtering & Management

- Gender filtering (Male, Female, Unisex, All)
- Origin region filtering (Nordic, European, International, etc.)
- Database-backed name storage
- **Git** submodule integration for name data updates
- Ratings persistence with **SQLite**

### Intelligent Pair Selection

- **Comparison tracking** - records every pairwise vote to understand name
  popularity
- **Phonetic similarity** - uses Double Metaphone algorithm to find phonetically
  similar names
- **Smart candidate selection** - prioritizes names with fewer comparisons and
  phonetically interesting pairs
- **Automatic metadata collection** - builds a feature dataset for Bayesian
  preference learning

### Active Learning with Bayesian Preference Modeling

- **Feature-based Bradley-Terry model** - replaces ELO with Bayesian preference
  learning using phonetic, linguistic, and metadata features
- **Thompson sampling** - selects maximally informative name pairs for human
  comparison
- **Multi-dimensional feature extraction** - uses **Double Metaphone** phonetic
  encoding, syllable counting, vowel ratios, gender, and origin features
- **Bayesian updates** - uses **Laplace approximation** for efficient posterior
  updates with each comparison
- **Uncertainty quantification** - maintains **covariance matrix** to model
  uncertainty in preferences
- **Database persistence** - stores model state in **SQLite** for consistent
  sessions
- **Preference scores** - uses Bayesian preference scores (1500 ± 500 scale) for
  UI display

### Performance Optimizations

- **2-second startup** - no automatic sync on app launch
- **Separated operations** - manual control over sync and classification
- **Batch processing** for origin classification (up to 100 names at a time)
- **Efficient database sync** with commit hash tracking
- **Bulk inserts** for new names
- **Selective logging** - suppresses 95% of debug noise from **watchdog** and **sqlite3**
- **Fallback mechanisms** for error recovery

## Quickstart

### Prerequisites

- **Python 3.13+** (required for modern type hints and features)
- **Git** (for submodule management)
- [uv](https://github.com/astral-sh/uv) - fast Python package manager

### Installation

1. **Clone the repository with submodules:**

   ```bash
   $ git clone --recurse-submodules https://github.com/yourusername/sort-names.git
   $ cd sort-names
   ```

2. **Install dependencies:**

   ```bash
   $ uv sync
   Resolved 87 packages in 0.42s
   Audited 87 packages in 0.01s
   ```

3. **Initialize the database:**

   ```bash
   $ uv run name-db init
   Database initialized successfully
   Synced 4,847 names from submodule
   ```

4. **Classify name origins (optional):**

   ```bash
   $ uv run name-db process
   Processing 100 names...
   Classified: 97 Nordic, 3 European
   ```

5. **Start the application:**

   ```bash
   $ uv run streamlit run st_name_ranking/main.py

     You can now view your Streamlit app in your browser.

     Local URL: http://localhost:8501
     Network URL: http://192.168.1.100:8501
   ```

The application runs at `http://localhost:8501`.

### Setup Details

The application uses **SQLite** for data storage. On first run, the database schema
creates automatically, but you populate it with names and optionally
run origin classification.

#### Setup Options

You have two approaches:

1. **UI‑Based Setup** (Recommended for first‑time users):

   - Run the application with `uv run streamlit run st_name_ranking/main.py`
   - Click **Sync Names** in the sidebar to load names from the submodule
   - Click **Classify Origins** to process name nationalities (optional)

2. **CLI‑Based Setup** (Advanced control):
   - Use the **Typer** CLI (as shown in the Installation steps)
   - Control batch sizes and incremental processing

#### What Happens Automatically:

- ✅ Database schema creation (tables, indexes)
- ✅ Default region mapping insertion
- ❌ Name sync from submodule (manual via UI or CLI)
- ❌ Origin classification (manual via UI or CLI)

### Basic Usage

1. **Tournament Mode**: Compare two names, vote for your preference
2. **Similarity Search**: Find names similar to a reference name
3. **Sidebar Controls**:
   - **Sync Names**: Update database from submodule
   - **Classify Origins**: Process name nationalities in batches
   - **Filters**: Filter by gender and origin region

## Development

### Project Structure

```
sort-names/
├── src/
│   └── st_name_ranking/
│       ├── main.py              # Streamlit application entry point
│       ├── database.py          # SQLite database operations (includes pairwise_comparisons and preference score constants)
│       ├── data_loader.py       # Name loading and validation
│       ├── similarity.py       # Name similarity functions (string & vector)
│       ├── phonetic_similarity.py # Double Metaphone phonetic matching
│       ├── features.py         # Feature extraction for active learning
│       ├── model.py            # Bayesian preference learning model
│       ├── ui.py              # Streamlit UI components
│       ├── utils.py           # Utility functions (smart candidate selection)
│       ├── classify_origins.py # Origin classification (ethnidata integration)
│       ├── init_database.py   # Database initialization
│       ├── cli.py            # Typer CLI for database management
│       ├── origin_classifier.py # Hierarchical origin classifier
│       └── __init__.py
├── scripts/                    # Utility scripts for batch classification, benchmarking, etc.
│   ├── benchmark_classification.py
│   ├── check_phonetic.py
│   ├── classify_1000.py
│   ├── classify_5000.py
│   ├── classify_all.py
│   ├── classify_remaining.py
│   ├── final_stats.py
│   └── test_classify.py
├── tests/                     # Test suite
├── docs/                      # Documentation
├── data/                      # Data directory
└── godkendtefornavne/         # Git submodule with Danish name data
```

### Testing

Run the test suite:

```bash
# Run all non-UI tests
$ uv run pytest -m "not playwright"
=================== test session starts ====================
platform linux -- Python 3.13.0
plugins: playwright-0.5.0
collected 47 items

47 passed in 3.24s
```

```bash
# Run specific test modules
$ uv run pytest tests/test_classify_origins.py -v
=================== test session starts ====================
tests/test_classify_origins.py::test_classify_name PASSED
tests/test_classify_origins.py::test_batch_classify PASSED
2 passed in 1.87s
```

```bash
$ uv run pytest tests/test_cli.py -v
=================== test session starts ====================
tests/test_cli.py::test_init_command PASSED
tests/test_cli.py::test_process_command PASSED
2 passed in 2.14s
```

#### Integration Tests (Playwright)

For end-to-end testing with a real browser, use **Playwright** integration tests:

```bash
# Install Playwright browsers (if not using Nix)
$ uv run playwright install chromium
✓ chromium downloaded to ~/.cache/ms-playwright/chromium-1145/
```

```bash
# Start the Streamlit application in another terminal:
$ uv run streamlit run src/st_name_ranking/main.py --server.port 8501
```

```bash
# Run integration tests (requires running application)
$ uv run pytest tests/test_integration_playwright.py --run-integration --run-playwright -v
=================== test session starts ====================
tests/test_integration_playwright.py::test_initial_load PASSED
tests/test_integration_playwright.py::test_voting_flow PASSED
2 passed in 8.42s
```

Note: When using **Nix**, browsers are provided via the `playwright-driver.browsers`
package. The integration tests automatically use the Nix-provided browsers.

### Code Quality

- Uses **ruff** for linting and formatting
- Follows **Python 3.13+** type hints
- Consistent code formatting with **ruff format**

### Generating Screenshots

For documentation purposes, generate screenshots of the application
using **Playwright**. First, ensure the application is running on
`http://localhost:8501`, then run:

```bash
$ uv run python scripts/take_screenshots.py
✓ Captured tournament_view.png (1200x800)
✓ Captured similarity_search.png (1200x800)
✓ Captured rankings_view.png (1200x800)
3 screenshots saved to docs/images/
```

This requires **Playwright** and browser binaries. See the script for installation
instructions.

```bash
# Format code
$ uv run ruff format .
84 files reformatted

# Lint code
$ uv run ruff check .
All checks passed
```

### Database Schema

The application uses **SQLite** (`names.db`) with the following schema:

```sql
-- Names table
CREATE TABLE names (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    gender TEXT CHECK(gender IN ('Male', 'Female', 'Unisex')),
    origin_region TEXT,
    origin_confidence REAL,
    origin_classified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ratings table
CREATE TABLE ratings (
    name_id INTEGER PRIMARY KEY REFERENCES names(id) ON DELETE CASCADE,
    rating REAL NOT NULL DEFAULT 1500.0,
    matches INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Region mapping table
CREATE TABLE region_mapping (
    nationality TEXT PRIMARY KEY,
    region TEXT NOT NULL
);

-- Source versions table (submodule tracking)
CREATE TABLE source_versions (
    id INTEGER PRIMARY KEY,
    commit_hash TEXT NOT NULL,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pairwise comparisons table (for tracking voting patterns)
CREATE TABLE pairwise_comparisons (
    id INTEGER PRIMARY KEY,
    name_a_id INTEGER NOT NULL REFERENCES names(id) ON DELETE CASCADE,
    name_b_id INTEGER NOT NULL REFERENCES names(id) ON DELETE CASCADE,
    winner_id INTEGER REFERENCES names(id) ON DELETE CASCADE,
    is_draw BOOLEAN DEFAULT FALSE,
    weight REAL DEFAULT 1.0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (name_a_id != name_b_id),
    CHECK (winner_id IS NULL OR winner_id IN (name_a_id, name_b_id)),
    CHECK (NOT (is_draw = TRUE AND winner_id IS NOT NULL))
);

-- Model state table (for active learning)
CREATE TABLE model_state (
    id INTEGER PRIMARY KEY,
    mean_weights BLOB NOT NULL,
    covariance BLOB NOT NULL,
    feature_names BLOB NOT NULL,
    feature_dim INTEGER NOT NULL,
    training_samples INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Comparisons table for preference learning
CREATE TABLE comparisons (
    id INTEGER PRIMARY KEY,
    name_a_id INTEGER NOT NULL REFERENCES names(id),
    name_b_id INTEGER NOT NULL REFERENCES names(id),
    preference INTEGER NOT NULL CHECK(preference IN (-1, 0, 1)),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name_a_id, name_b_id, preference)
);
```

### Origin Classification

The application uses **ethnidata** for name nationality prediction. Control
classification via:

**Web Interface:**

- Auto-classify after update checkbox
- Classify 100 Names (incremental)
- Classify All (full batch)

**Command Line:**

```bash
$ uv run name-db process --limit 100    # Test with 100 names
Processing 100 names...
Classified: 97 Nordic, 3 European
Completed in 3.2s

$ uv run name-db process --batch-size 50 # Custom batch size
Processing 50 names...
Completed in 1.8s

$ uv run name-db process                 # Classify all names
Processing 4,847 names...
Classified: 4,623 Nordic, 224 European
Completed in 87.4s
```

## Detailed Usage

### CLI Reference

The **Typer** CLI provides comprehensive database management:

```bash
# Show all commands
$ uv run name-db --help

 Usage: name-db [OPTIONS] COMMAND [ARGS]...

 Commands:
   init         Initialize database
   process      Process data enrichment
   stats        Show statistics
   model-status Show active learning model status
   model-reset  Reset active learning model
```

```bash
# Initialize database (schema + sync names + optional classification)
$ uv run name-db init [--classify]
Database initialized successfully
Synced 4,847 names from submodule
```

```bash
# Process data enrichment (origin classification)
$ uv run name-db process [--limit N] [--batch-size N]
Processing 100 names...
Classified: 97 Nordic, 3 European
```

```bash
# Show database statistics
$ uv run name-db stats
Total names: 4,847
Classified: 4,623 (95.4%)
Comparisons: 1,247
```

```bash
# Show active learning model status
$ uv run name-db model-status
Model trained: Yes
Training samples: 1,247
Feature dimensions: 15
```

```bash
# Reset active learning model
$ uv run name-db model-reset
Model state cleared successfully
```

**Note**: The `init` command automatically syncs names from the submodule. There
are no separate `sync` or `migrate` commands. To re-sync names, run
`init` again (it's idempotent for schema creation).

### Web Application Workflow

1. **2-Second Startup**: Application loads with automatic schema
   initialization (no automatic name sync)
2. **Manual Control**: Separate buttons for reload, sync, and git updates
3. **Incremental Processing**: Classify names in batches of 100
4. **Real-time Progress**: See classification percentage as it processes

### Performance Optimizations

- **Batch processing**: Reduces **ethnidata** API calls by 100x
- **Efficient sync**: Commit hash tracking avoids redundant processing
- **Bulk inserts**: `executemany` for fast database operations
- **Selective logging**: Suppresses 95% of debug noise for cleaner output
- **UI responsiveness**: Batch database queries, feature caching, and
  pre-fetching for instant pair selection
- **Vectorized computations**: **NumPy**-optimized pair scoring for active learning
- **Candidate queue**: Pre-fetches 3 name pairs to eliminate selection latency

## Documentation

For detailed technical documentation, including active learning system design,
architecture, and implementation details, see the [documentation](./docs/)
directory.

## License

[Add your license here]

## Acknowledgments

- [**ethnidata**](https://github.com/teyfikoz/ethnidata) for name nationality
  prediction
- **Streamlit** for the web application framework
- The Danish government for the name data
