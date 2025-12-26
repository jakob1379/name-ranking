# Name Ranking Application

A Streamlit-based web application for ranking Danish names using Elo rating system, with similarity search and origin classification.

## ✨ Features

### 🏆 Elo Rating Tournament
- Compare two random names side-by-side
- Vote for preferred name or mark as draw
- Real-time Elo rating updates
- Top 10 rankings display
- Keyboard shortcuts (← → ↑ arrows)

### 🔍 Similarity Search
- **String similarity** (Levenshtein distance) for name matching
- **Vector similarity** (LLM embeddings) for semantic matching
- Find names similar to a reference name

### 🌍 Origin Classification
- **Optional classification** - runs only when explicitly requested
- **Incremental processing** - classify 100 names at a time or all at once
- Automatic nationality prediction using `ethnidata`
- Mapping to geographic regions (Nordic, European, Asian, etc.)
- Confidence scoring for predictions
- Batch processing for unclassified names
- **Progress tracking** - shows classification percentage in UI

### ⚙️ Filtering & Management
- Gender filtering (Male, Female, Unisex, All)
- Origin region filtering (Nordic, European, International, etc.)
- Database-backed name storage
- Git submodule integration for name data updates
- Ratings persistence with SQLite

### 📊 Performance Optimizations
- **Fast startup** - no automatic sync on app launch
- **Separated operations** - manual control over sync and classification
- **Batch processing** for origin classification (up to 100 names at a time)
- **Efficient database sync** with commit hash tracking
- **Bulk inserts** for new names
- **Selective logging** - suppresses debug noise from watchdog and sqlite3
- **Fallback mechanisms** for error recovery

## 🚀 Quickstart

### Prerequisites
- Python 3.13 or higher
- Git (for submodule management)
- [uv](https://github.com/astral-sh/uv) - fast Python package manager

### Installation
1. **Clone the repository with submodules:**
   ```bash
   git clone --recurse-submodules https://github.com/yourusername/sort-names.git
   cd sort-names
   ```

2. **Set up the environment and install dependencies:**
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

### Database Setup
Initialize the database with names and optional origin classification:

```bash
# Initialize database (schema + sync names + migrate ratings)
uv run name-db init

# Initialize with origin classification
uv run name-db init --classify

# Or use individual commands for more control:
uv run name-db sync      # Sync names from submodule
uv run name-db migrate   # Migrate ratings from ratings.json
uv run name-db classify  # Classify name origins
uv run name-db stats     # Show database statistics
```

### Running the Application
Start the Streamlit web application:

```bash
uv run streamlit run st_name_ranking/main.py
```

The application will be available at `http://localhost:8501`.

### Basic Usage
1. **Tournament Mode**: Compare two names, vote for your preference
2. **Similarity Search**: Find names similar to a reference name
3. **Sidebar Controls**:
   - **Sync Names**: Update database from submodule
   - **Classify Origins**: Process name nationalities in batches
   - **Filters**: Filter by gender and origin region

## 🛠️ Development

### Project Structure
```
st_name_ranking/
├── main.py              # Streamlit application entry point
├── database.py          # SQLite database operations
├── data_loader.py       # Name loading and validation
├── elo.py              # Elo rating calculations
├── similarity.py       # Name similarity functions
├── ui.py              # Streamlit UI components
├── utils.py           # Utility functions
├── classify_origins.py # Origin classification (ethnidata integration)
├── init_database.py   # Database initialization
├── cli.py            # Typer CLI for database management
└── __init__.py
```

### Testing
Run the test suite:

```bash
# Run all non-UI tests
uv run pytest -m "not playwright"

# Run specific test modules
uv run pytest tests/test_classify_origins.py
uv run pytest tests/test_cli.py
```

### Code Quality
- Uses `ruff` for linting and formatting
- Follows Python type hints
- Consistent code formatting with `ruff format`

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

### Database Schema
The application uses SQLite (`names.db`) with the following schema:

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
```

### Origin Classification
The application uses `ethnidata` for name nationality prediction. Classification can be controlled via:

**Web Interface:**
- Auto-classify after update checkbox
- Classify 100 Names (incremental)
- Classify All (full batch)

**Command Line:**
```bash
uv run name-db classify --limit 100    # Test with 100 names
uv run name-db classify --batch-size 50 # Custom batch size
uv run name-db classify                 # Classify all names
```

## 📖 Detailed Usage

### CLI Reference
The `name-db` CLI provides comprehensive database management:

```bash
# Show all commands
uv run name-db --help

# Initialize database with options
uv run name-db init [--classify] [--ratings-path PATH]

# Sync names from submodule
uv run name-db sync

# Migrate ratings from JSON file
uv run name-db migrate [--ratings-path PATH]

# Classify name origins
uv run name-db classify [--limit N] [--batch-size N]

# Show database statistics
uv run name-db stats
```

### Web Application Workflow
1. **Fast Startup**: Application loads instantly without automatic sync
2. **Manual Control**: Separate buttons for reload, sync, and git updates
3. **Incremental Processing**: Classify names in batches of 100
4. **Real-time Progress**: See classification percentage as it processes

### Performance Optimizations
- **Batch processing**: Reduces `ethnidata` API calls by ~100x
- **Efficient sync**: Commit hash tracking avoids redundant processing
- **Bulk inserts**: `executemany` for fast database operations
- **Selective logging**: Suppresses debug noise for cleaner output

## 📝 License
[Add your license here]

## 🙏 Acknowledgments
- [ethnidata](https://github.com/teyfikoz/ethnidata) for name nationality prediction
- Streamlit for the web application framework
- The Danish government for the name data
