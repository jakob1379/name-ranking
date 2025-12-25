# Name Ranking Application

A Streamlit-based web application for ranking Danish names using Elo rating system, with similarity search and origin classification.

## Features

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
- Automatic nationality prediction using `name2nat`
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
- **Batch processing** for origin classification (up to 100 names at once)
- **Efficient database sync** with commit hash tracking
- **Bulk inserts** for new names
- **Selective logging** - suppresses debug noise from watchdog and sqlite3
- **Fallback mechanisms** for error recovery

## Setup

### Prerequisites
- Python 3.13 or higher
- Git (for submodule management)

### Installation

1. **Clone the repository with submodules:**
   ```bash
   git clone --recurse-submodules https://github.com/yourusername/sort-names.git
   cd sort-names
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

4. **Optional: Install name2nat for origin classification:**
   ```bash
   pip install name2nat
   ```

### Database Initialization

The application uses SQLite (`names.db`). The database is automatically initialized on first run.

#### Using the CLI (Recommended)

A Typer-based CLI provides comprehensive database management via installed scripts:

```bash
# Show all available commands
name-db --help
# or if not in PATH: uv run name-db --help

# Initialize database (schema + sync + migrate)
name-db init

# Initialize with origin classification
name-db init --classify

# Sync names from submodule
name-db sync

# Migrate ratings from JSON
name-db migrate

# Classify name origins
name-db classify --limit 100  # test with 100 names
name-db classify              # classify all names

# Show database statistics
name-db stats
```

Additional installed scripts:
```bash
# Direct script access (legacy compatibility)
init-db --help
classify-origins --help
```

#### Using Legacy Scripts

The original scripts are still available inside the package:

```bash
# Initialize database
python -m st_name_ranking.init_database --classify

# Classify origins
python -m st_name_ranking.classify_origins --limit 100 --stats
```

## Usage

### Running the Application

Start the Streamlit app:
```bash
streamlit run st_name_ranking/main.py
```

The application will be available at `http://localhost:8501`.

### Application Interface

1. **Sidebar**
   - **Submodule Management** (three separate controls):
     - **Reload Names**: Fast reload from existing database
     - **Sync Names**: Sync database with local submodule (no git pull)
     - **Check for Updates**: Pull git updates and sync (optional auto-classification)
   - **Origin Classification**:
     - **Auto-classify after update** checkbox
     - **Classify 100 Names** button (incremental processing)
     - **Classify All** button (full batch processing)
     - Progress display showing `X/Y names (Z%)`
   - Gender filter selection
   - Origin region filter (multiselect)
   - Ratings management (save, reset, export)

2. **Main Content**
   - **Tournament tab**: Compare two names, vote, see top rankings
   - **Similarity tab**: Search for names similar to a reference

### Optimized Workflow

The application has been optimized for faster startup and better user control:

1. **Fast Startup**: App loads instantly from existing database without automatic sync
2. **Manual Sync Control**: Three separate buttons for different sync scenarios:
   - **Reload Names**: Quick refresh from database
   - **Sync Names**: Sync with local submodule changes
   - **Check for Updates**: Pull git updates and sync
3. **Incremental Classification**: Process 100 names at a time or all at once
4. **Progress Visibility**: Real-time tracking of classification progress

### Origin Classification

Origin classification can be done in two ways:

#### 1. Via the Web Interface (Recommended)
- **Auto-classify after update**: Checkbox to automatically classify after git updates
- **Classify 100 Names**: Incremental processing for testing or slow systems
- **Classify All**: Full batch processing (may take several minutes)
- **Progress tracking**: Shows real-time classification percentage

#### 2. Via Command Line
```bash
classify-origins
# or: name-db classify
```

Options:
- `--limit N`: Limit to N names (for testing)
- `--batch-size N`: Set batch size (default: 100)
- `--stats`: Show classification statistics

### Database Management

- Names are stored in `names.db` SQLite database
- Ratings are persisted automatically
- **Manual sync control** - user decides when to sync with submodule
- Submodule updates are tracked by commit hash to avoid redundant processing
- Unclassified names can be processed in batches (100 at a time or all at once)

## Optimization Details

### Batch Processing
- Origin classification processes up to 100 names simultaneously
- Reduces API calls to `name2nat` by ~100x
- Fallback to individual classification if batch fails

### Database Efficiency
- Bulk inserts using `executemany` for new names
- Unique constraints prevent duplicates
- Indexed columns for fast filtering
- Commit hash tracking avoids redundant processing

### Logging System
- Structured logging with timestamps and module names
- **Suppressed debug noise** - watchdog and sqlite3 loggers set to WARNING level
- INFO level: Milestones (database initialized, names loaded, ratings saved)
- WARNING level: Potential issues (missing files, classification failures)
- ERROR level: Critical failures

### Error Handling
- Graceful fallbacks for missing dependencies
- Individual name classification continues if batch fails
- Database transactions ensure data consistency

## Database Schema

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

-- User settings table
CREATE TABLE user_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

## Development

### Code Structure
- `main.py` - Streamlit application entry point
- `database.py` - SQLite database operations
- `data_loader.py` - Name loading and validation
- `elo.py` - Elo rating calculations
- `similarity.py` - Name similarity functions
- `ui.py` - Streamlit UI components
- `utils.py` - Utility functions
- `classify_origins.py` - Origin classification script
- `init_database.py` - Database initialization script

### Testing
Run basic functionality tests:
```bash
python verify_imports.py
```

### Code Quality
- Uses `ruff` for linting
- Follows Python type hints
- Consistent code formatting

## Data Sources

Names are sourced from the `godkendtefornavne` submodule, which contains approved Danish names from the Danish government.

## License

[Add your license here]

## Acknowledgments

- [name2nat](https://github.com/ivarvit/name2nat) for name nationality prediction
- Streamlit for the web application framework
- The Danish government for the name data
